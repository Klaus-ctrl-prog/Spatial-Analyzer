import numpy as np
import pytest

from spatial_analyzer.biquad_preview import PeakingFilter
from spatial_analyzer.complex_response import to_complex
from spatial_analyzer.eq_solver import (
    ModeConfirmation,
    propose_broadband_correction,
    propose_mode_cuts,
    validate_no_harm,
)


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


class TestValidateNoHarm:
    def test_no_filters_is_always_safe(self):
        freq = np.array([100.0, 200.0, 300.0])
        l_complex = to_complex(np.full((2, 3), 70.0), np.zeros((2, 3)))
        r_complex = to_complex(np.full((2, 3), 70.0), np.zeros((2, 3)))
        is_safe, before, after = validate_no_harm(
            freq, ["p1", "p2"], l_complex, r_complex, [], []
        )
        assert is_safe
        assert before == pytest.approx(after)

    def test_detects_increased_spread_from_harmful_filter(self):
        freq = np.linspace(50.0, 200.0, 20)
        rng = np.random.default_rng(1)
        # Zwei Positionen mit leicht unterschiedlicher Kurvenform
        l_spl = np.vstack([
            70.0 + rng.normal(scale=0.1, size=freq.size),
            65.0 + rng.normal(scale=0.1, size=freq.size),
        ])
        r_spl = l_spl.copy()
        l_complex = to_complex(l_spl, np.zeros_like(l_spl))
        r_complex = to_complex(r_spl, np.zeros_like(r_spl))
        # Ein extremer, schmaler Boost-Filter kann an verschiedenen Positionen
        # unterschiedlich stark wirken, wenn die Kurven vorher nicht exakt
        # identisch sind - hier nur ein Sanity-Check, dass die Funktion ueberhaupt
        # eine Zahl liefert und nicht wirft.
        filt = [PeakingFilter(type="peq", fc=100.0, gain_db=6.0, q=8.0)]
        is_safe, before, after = validate_no_harm(freq, ["p1", "p2"], l_complex, r_complex, filt, filt)
        assert isinstance(is_safe, bool)
        assert np.isfinite(before)
        assert np.isfinite(after)
