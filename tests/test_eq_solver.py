import numpy as np
import pytest

from spatial_analyzer.biquad_preview import PeakingFilter
from spatial_analyzer.complex_response import to_complex
from spatial_analyzer.eq_solver import (
    CHECKED_SCOPE_LEFT_RIGHT,
    CHECKED_SCOPE_NONE,
    ModeConfirmation,
    propose_broadband_correction,
    propose_mode_cuts,
    NoHarmStatus,
    validate_no_harm,
)
from spatial_analyzer.filter_digest import peaking_filters_digest


def _flat_curve(freq, level_db):
    return np.full_like(freq, level_db, dtype=np.float64)


class TestProposeModeCuts:
    def test_confirmed_mode_above_target_gets_cut_only(self):
        freq = np.linspace(20.0, 200.0, 50)
        averaged = _flat_curve(freq, 65.0)
        idx = np.argmin(np.abs(freq - 60.0))
        averaged[idx] = 75.0  # +10dB Peak
        target = _flat_curve(freq, 65.0)
        confirmations = [
            ModeConfirmation(freq_hz=60.0, classification="mode_candidate", decay_confirmed=True, q_factor=5.0)
        ]
        proposals = propose_mode_cuts(freq, averaged, target, confirmations, max_cut_db=9.0)
        assert len(proposals) == 1
        assert proposals[0].filter.gain_db < 0
        assert proposals[0].filter.gain_db == pytest.approx(-9.0)  # auf max_cut_db begrenzt

    def test_unconfirmed_decay_is_skipped(self):
        freq = np.linspace(20.0, 200.0, 50)
        averaged = _flat_curve(freq, 75.0)
        target = _flat_curve(freq, 65.0)
        confirmations = [
            ModeConfirmation(freq_hz=60.0, classification="mode_candidate", decay_confirmed=False)
        ]
        proposals = propose_mode_cuts(freq, averaged, target, confirmations)
        assert proposals == []

    def test_comb_filter_candidate_is_never_cut(self):
        freq = np.linspace(20.0, 200.0, 50)
        averaged = _flat_curve(freq, 75.0)
        target = _flat_curve(freq, 65.0)
        confirmations = [
            ModeConfirmation(freq_hz=60.0, classification="comb_filter_candidate", decay_confirmed=True)
        ]
        proposals = propose_mode_cuts(freq, averaged, target, confirmations)
        assert proposals == []

    def test_mode_already_below_target_is_skipped(self):
        freq = np.linspace(20.0, 200.0, 50)
        averaged = _flat_curve(freq, 60.0)
        target = _flat_curve(freq, 65.0)
        confirmations = [
            ModeConfirmation(freq_hz=60.0, classification="mode_candidate", decay_confirmed=True)
        ]
        proposals = propose_mode_cuts(freq, averaged, target, confirmations)
        assert proposals == []

    def test_outside_band_is_ignored(self):
        freq = np.linspace(20.0, 300.0, 50)
        averaged = _flat_curve(freq, 75.0)
        target = _flat_curve(freq, 65.0)
        confirmations = [
            ModeConfirmation(freq_hz=250.0, classification="mode_candidate", decay_confirmed=True)
        ]
        proposals = propose_mode_cuts(freq, averaged, target, confirmations, band_hz=(20.0, 200.0))
        assert proposals == []


class TestProposeBroadbandCorrection:
    def test_proposes_cut_for_positive_deviation(self):
        freq = np.linspace(200.0, 5000.0, 200)
        averaged = _flat_curve(freq, 70.0)
        idx = np.argmin(np.abs(freq - 1000.0))
        averaged[idx] = 76.0
        target = _flat_curve(freq, 70.0)
        proposals = propose_broadband_correction(freq, averaged, target, max_q=2.0, max_cut_db=6.0)
        assert len(proposals) >= 1
        closest = min(proposals, key=lambda p: abs(p.filter.fc - 1000.0))
        assert closest.filter.gain_db < 0
        assert closest.filter.q <= 2.0

    def test_excluded_frequency_is_never_proposed(self):
        freq = np.linspace(200.0, 5000.0, 200)
        averaged = _flat_curve(freq, 70.0)
        idx = np.argmin(np.abs(freq - 1000.0))
        averaged[idx] = 76.0
        target = _flat_curve(freq, 70.0)
        proposals = propose_broadband_correction(
            freq, averaged, target, excluded_freqs_hz=[float(freq[idx])]
        )
        assert all(abs(p.filter.fc - 1000.0) > 1.0 for p in proposals)

    def test_small_deviation_below_threshold_proposes_nothing(self):
        freq = np.linspace(200.0, 5000.0, 50)
        averaged = _flat_curve(freq, 70.1)
        target = _flat_curve(freq, 70.0)
        proposals = propose_broadband_correction(freq, averaged, target, deviation_threshold_db=1.0)
        assert proposals == []

    def test_respects_max_filters_limit(self):
        rng = np.random.default_rng(0)
        freq = np.linspace(200.0, 10000.0, 500)
        averaged = 70.0 + rng.normal(scale=5.0, size=freq.size)
        target = _flat_curve(freq, 70.0)
        proposals = propose_broadband_correction(
            freq, averaged, target, max_filters=3, deviation_threshold_db=0.5, min_fc_spacing_oct=0.1
        )
        assert len(proposals) <= 3


def _near_cancellation_setup(freq, phase_offsets_deg=(0.0, -5.0, 5.0)):
    """Drei Positionen, L=1, R=0.5 mit ~180 Grad Phasenversatz (leichte
    positionsabhaengige Abweichung): die Summe liegt nahe einer
    Teilausloeschung und ist daher extrem empfindlich gegen eine
    kanalgetrennte Pegelaenderung von R."""
    n = len(phase_offsets_deg)
    l_complex = np.ones((n, freq.size), dtype=np.complex128)
    r_phase = np.deg2rad(180.0 + np.asarray(phase_offsets_deg))[:, None]
    r_complex = 0.5 * np.exp(1j * r_phase) * np.ones((n, freq.size))
    return l_complex, r_complex


class TestValidateNoHarm:
    def test_no_filters_is_safe_even_with_one_position(self):
        freq = np.array([100.0, 200.0, 300.0])
        l_complex = to_complex(np.full((1, 3), 70.0), np.zeros((1, 3)))
        result = validate_no_harm(freq, ["p1"], l_complex, l_complex, [], [])
        assert result.status is NoHarmStatus.SAFE
        assert result.is_released

    @pytest.mark.parametrize("n", [1, 2])
    def test_too_few_positions_is_inconclusive_not_silent_nan(self, n):
        freq = np.linspace(90.0, 110.0, 21)
        l_complex, r_complex = _near_cancellation_setup(freq, (0.0, -5.0)[:n])
        filt = [PeakingFilter(type="peq", fc=100.0, gain_db=6.0, q=1.0)]
        result = validate_no_harm(freq, [f"p{i}" for i in range(n)], l_complex, r_complex, [], filt)
        assert result.status is NoHarmStatus.INCONCLUSIVE
        assert not result.is_released
        assert result.n_positions == n
        assert result.spread_before_db is None  # kein stiller NaN-Wert

    def test_different_lr_filters_increase_spread_is_unsafe(self):
        freq = np.linspace(90.0, 110.0, 21)
        l_complex, r_complex = _near_cancellation_setup(freq)
        boost_r = [PeakingFilter(type="peq", fc=100.0, gain_db=6.0, q=0.5)]
        result = validate_no_harm(freq, ["p1", "p2", "p3"], l_complex, r_complex, [], boost_r)
        assert result.status is NoHarmStatus.UNSAFE
        assert result.spread_after_db > result.spread_before_db + 1.0
        assert not result.is_released

    def test_identical_lr_filters_leave_spread_unchanged_and_safe(self):
        freq = np.linspace(90.0, 110.0, 21)
        l_complex, r_complex = _near_cancellation_setup(freq)
        filt = [PeakingFilter(type="peq", fc=100.0, gain_db=-4.0, q=1.0)]
        result = validate_no_harm(freq, ["p1", "p2", "p3"], l_complex, r_complex, filt, filt)
        assert result.status is NoHarmStatus.SAFE
        assert result.spread_after_db == pytest.approx(result.spread_before_db, abs=1e-9)

    def test_narrowband_resonance_is_caught_by_local_criterion(self):
        # Schmalbandiger Boost auf R: der globale Mittelwert bleibt unter der
        # globalen Toleranz, die Streuung steigt aber an EINER Frequenz stark.
        freq = np.geomspace(20.0, 200.0, 400)
        l_complex, r_complex = _near_cancellation_setup(freq)
        narrow = [PeakingFilter(type="peq", fc=100.0, gain_db=6.0, q=20.0)]
        result = validate_no_harm(freq, ["p1", "p2", "p3"], l_complex, r_complex, [], narrow)
        assert result.spread_after_db - result.spread_before_db <= 0.5  # global unauffaellig
        assert result.max_local_spread_increase_db > 2.0
        assert result.max_local_spread_increase_hz == pytest.approx(100.0, rel=0.1)
        assert result.status is NoHarmStatus.UNSAFE
        assert "lokal" in result.reason

    def test_target_deviation_increase_is_unsafe(self):
        freq = np.linspace(50.0, 200.0, 40)
        rng = np.random.default_rng(3)
        spl = 70.0 + rng.normal(scale=0.2, size=(3, freq.size))
        l_complex = to_complex(spl, np.zeros_like(spl))
        r_complex = l_complex.copy()
        boost = [PeakingFilter(type="peq", fc=100.0, gain_db=6.0, q=0.3)]
        target = np.full(freq.shape, 76.02)  # = Summenpegel 70 dB + 6.02 dB vor dem Filter
        without_target = validate_no_harm(freq, ["a", "b", "c"], l_complex, r_complex, boost, boost)
        with_target = validate_no_harm(
            freq, ["a", "b", "c"], l_complex, r_complex, boost, boost, target_db=target - 6.02
        )
        assert without_target.status is NoHarmStatus.SAFE  # Streuung bleibt gleich
        assert with_target.status is NoHarmStatus.UNSAFE
        assert with_target.target_dev_after_db > with_target.target_dev_before_db
        assert "Zielkurve" in with_target.reason


class TestNoHarmBinding:
    """Freigabebindung (Package A): validate_no_harm meldet Umfang und Digest der geprueften Filter."""

    @staticmethod
    def _setup():
        freq = np.linspace(90.0, 110.0, 21)
        l_complex, r_complex = _near_cancellation_setup(freq)
        return freq, l_complex, r_complex

    def test_safe_run_reports_left_right_scope_and_digest(self):
        """Regression: ohne Scope/Digest konnte ein SAFE-Ergebnis ohne reale Pruefung wie eine Pruefung wirken."""
        freq, l_complex, r_complex = self._setup()
        filt = [PeakingFilter(type="peq", fc=100.0, gain_db=-4.0, q=1.0)]
        result = validate_no_harm(freq, ["p1", "p2", "p3"], l_complex, r_complex, filt, filt)
        assert result.status is NoHarmStatus.SAFE
        assert result.checked_scope == CHECKED_SCOPE_LEFT_RIGHT
        assert result.checked_filters_sha256 == peaking_filters_digest(filt, filt)

    def test_unsafe_run_also_reports_scope_and_digest_of_checked_filters(self):
        """Regression: UNSAFE-Ergebnisse tragen denselben Bezug zu den geprueften Filtern (Status unveraendert)."""
        freq, l_complex, r_complex = self._setup()
        boost_r = [PeakingFilter(type="peq", fc=100.0, gain_db=6.0, q=0.5)]
        result = validate_no_harm(freq, ["p1", "p2", "p3"], l_complex, r_complex, [], boost_r)
        assert result.status is NoHarmStatus.UNSAFE
        assert result.checked_scope == CHECKED_SCOPE_LEFT_RIGHT
        assert result.checked_filters_sha256 == peaking_filters_digest([], boost_r)

    def test_no_filters_has_scope_none_and_no_digest(self):
        """Regression: 'keine Filter' darf keinen Pruefumfang vortaeuschen."""
        freq = np.array([100.0, 200.0, 300.0])
        l_complex = to_complex(np.full((1, 3), 70.0), np.zeros((1, 3)))
        result = validate_no_harm(freq, ["p1"], l_complex, l_complex, [], [])
        assert result.status is NoHarmStatus.SAFE
        assert result.checked_scope == CHECKED_SCOPE_NONE
        assert result.checked_filters_sha256 is None

    def test_inconclusive_has_scope_none_and_no_digest(self):
        """Regression: eine nicht abgeschlossene Pruefung (zu wenige Positionen) ist nicht als geprueft markiert."""
        freq, l_complex, r_complex = self._setup()
        l2, r2 = l_complex[:2], r_complex[:2]
        filt = [PeakingFilter(type="peq", fc=100.0, gain_db=6.0, q=1.0)]
        result = validate_no_harm(freq, ["p1", "p2"], l2, r2, filt, filt)
        assert result.status is NoHarmStatus.INCONCLUSIVE
        assert result.checked_scope == CHECKED_SCOPE_NONE
        assert result.checked_filters_sha256 is None

    def test_status_and_reason_unchanged_for_unsafe_case(self):
        """Die Do-No-Harm-Entscheidung bleibt unveraendert (keine Aenderung der Logik durch die Bindung)."""
        freq, l_complex, r_complex = self._setup()
        boost_r = [PeakingFilter(type="peq", fc=100.0, gain_db=6.0, q=0.5)]
        result = validate_no_harm(freq, ["p1", "p2", "p3"], l_complex, r_complex, [], boost_r)
        assert result.status is NoHarmStatus.UNSAFE and not result.is_released
        assert "Streuung" in result.reason
        assert result.n_positions == 3

    def test_dataclass_positional_construction_still_works(self):
        """Bestehende positionale Aufrufe von NoHarmResult bleiben gueltig; neue Felder haben Defaults."""
        from spatial_analyzer.eq_solver import NoHarmResult

        res = NoHarmResult(NoHarmStatus.SAFE, "ok", 5)
        assert res.checked_scope == CHECKED_SCOPE_NONE and res.checked_filters_sha256 is None

    def test_checked_fields_cannot_be_passed_to_constructor(self):
        """Regression N1: Pruefnachweis laesst sich nicht per Konstruktor faelschen (init=False)."""
        from spatial_analyzer.eq_solver import NoHarmResult

        with pytest.raises(TypeError):
            NoHarmResult(NoHarmStatus.SAFE, "fake", 0, checked_scope="left_right", checked_filters_sha256="a" * 64)
        with pytest.raises(TypeError):
            NoHarmResult(NoHarmStatus.SAFE, "fake", 5, checked_scope="left_right")

    def test_result_is_frozen_after_creation(self):
        """Regression N1: validate_no_harm-Ergebnis laesst sich nicht nachtraeglich umschreiben (frozen)."""
        import dataclasses

        freq, l_complex, r_complex = self._setup()
        filt = [PeakingFilter(type="peq", fc=100.0, gain_db=-4.0, q=1.0)]
        result = validate_no_harm(freq, ["p1", "p2", "p3"], l_complex, r_complex, filt, filt)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.checked_scope = CHECKED_SCOPE_NONE
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.status = NoHarmStatus.UNSAFE
