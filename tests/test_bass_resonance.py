import numpy as np
import pytest

from spatial_analyzer.bass_resonance import (
    compare_to_room_model,
    evaluate_decay_against_threshold,
    find_peak_candidates,
    narrowband_decay_time,
    perceptual_threshold_s,
    spatial_consistency,
)
from spatial_analyzer.models import FieldMeaning, RoomModel


class TestFindPeakCandidates:
    def test_detects_synthetic_peak(self):
        freq = np.linspace(20.0, 200.0, 200)
        spl = np.full_like(freq, 60.0)
        # Lorentz-artiger Peak um 60 Hz
        spl += 10.0 / (1.0 + ((freq - 60.0) / 3.0) ** 2)
        candidates = find_peak_candidates(freq, spl, prominence_db=3.0)
        assert len(candidates) >= 1
        closest = min(candidates, key=lambda c: abs(c.freq_hz - 60.0))
        assert abs(closest.freq_hz - 60.0) < 5.0
        assert closest.q_factor is not None
        assert closest.q_factor > 0

    def test_flat_spectrum_has_no_peaks(self):
        freq = np.linspace(20.0, 200.0, 100)
        spl = np.full_like(freq, 65.0)
        candidates = find_peak_candidates(freq, spl)
        assert candidates == []

    def test_too_few_points_in_band_returns_empty(self):
        freq = np.array([20.0, 500.0])
        spl = np.array([60.0, 61.0])
        candidates = find_peak_candidates(freq, spl, band_hz=(20.0, 200.0))
        assert candidates == []


class TestSpatialConsistency:
    def test_consistent_cluster_across_all_positions(self):
        peaks = [[60.0], [60.2], [59.8], [60.1]]
        clusters = spatial_consistency(peaks, tolerance_hz=2.0)
        assert len(clusters) == 1
        freq, fraction = clusters[0]
        assert fraction == pytest.approx(1.0)
        assert abs(freq - 60.0) < 1.0

    def test_inconsistent_scattered_peaks_form_separate_clusters(self):
        peaks = [[60.0], [120.0], [45.0], [200.0]]
        clusters = spatial_consistency(peaks, tolerance_hz=2.0)
        assert len(clusters) == 4
        for _, fraction in clusters:
            assert fraction == pytest.approx(0.25)

    def test_empty_input_gives_empty_result(self):
        assert spatial_consistency([[], [], []]) == []


class TestCompareToRoomModel:
    def test_matches_known_candidate(self):
        room = RoomModel(length_m=5.72, width_m=3.72, height_m=2.40)
        result = compare_to_room_model(60.0, room, tolerance_hz=2.0)
        assert result is not None
        key, freq, meaning = result
        assert key == "length_2"
        assert meaning == FieldMeaning.MODEL

    def test_no_candidate_within_tolerance_returns_none(self):
        room = RoomModel(length_m=5.72, width_m=3.72, height_m=2.40)
        result = compare_to_room_model(150.0, room, tolerance_hz=1.0)
        assert result is None


class TestNarrowbandDecayTime:
    def test_exponential_decay_recovered_approximately(self):
        fs = 4000.0
        dt = 1.0 / fs
        n = 8000
        t = np.arange(n) * dt
        f0 = 60.0
        true_tau_60db = 0.5  # Sekunden bis -60dB
        slope_db_per_s = -60.0 / true_tau_60db
        envelope_db = slope_db_per_s * t
        amplitude = 10 ** (envelope_db / 20.0)
        ir = amplitude * np.sin(2 * np.pi * f0 * t)
        decay = narrowband_decay_time(ir, dt, center_freq_hz=f0, bandwidth_oct=1.0)
        assert decay == pytest.approx(true_tau_60db, rel=0.3)

    def test_raises_on_invalid_band(self):
        ir = np.zeros(100)
        with pytest.raises(ValueError):
            narrowband_decay_time(ir, 1.0, center_freq_hz=1.0, bandwidth_oct=0.01)


class TestPerceptualThreshold:
    """Haertung (ab 2026-11-22, schliesst Issue #2): Stuetzpunkte jetzt
    direkt aus Fazenda, Stephenson & Goldberg (2015), Abschnitt
    "Conclusion", "artificial stimuli"-Mittelwerte (siehe Modul-Docstring
    fuer die genaue Zitatstelle)."""

    def test_monotonic_decreasing_with_frequency(self):
        values = [perceptual_threshold_s(f) for f in (20.0, 50.0, 100.0, 200.0)]
        assert all(earlier > later for earlier, later in zip(values, values[1:]))

    @pytest.mark.parametrize(
        "freq_hz,expected_s",
        [(32.0, 0.90), (63.0, 0.30), (100.0, 0.27), (150.0, 0.18), (200.0, 0.17)],
    )
    def test_exact_support_points_match_published_values(self, freq_hz, expected_s):
        # Direkt aus dem Fliesstext der Publikation uebernommene Werte -
        # siehe Modul-Docstring fuer die Zitatstelle (Abschnitt "Conclusion").
        assert perceptual_threshold_s(freq_hz) == pytest.approx(expected_s, abs=1e-9)

    def test_interpolated_point_matches_log_log_linear_formula(self):
        # Unabhaengig nachgerechneter Punkt zwischen zwei Stuetzpunkten
        # (63 Hz / 0.30s und 100 Hz / 0.27s), um die dokumentierte
        # Interpolationsform (linear in log10(f) UND log10(t)) direkt zu
        # pruefen, nicht nur die Monotonie.
        f_lo, t_lo = 63.0, 0.30
        f_hi, t_hi = 100.0, 0.27
        f_mid = 80.0
        frac = (np.log10(f_mid) - np.log10(f_lo)) / (np.log10(f_hi) - np.log10(f_lo))
        expected = 10.0 ** (np.log10(t_lo) + frac * (np.log10(t_hi) - np.log10(t_lo)))
        assert perceptual_threshold_s(f_mid) == pytest.approx(expected, rel=1e-9)

    def test_clamped_below_lowest_support_point_not_extrapolated(self):
        # Die Studie deckt nur 32-250 Hz ab (artificial stimuli: 32-200 Hz)
        # - unterhalb von 32 Hz wird bewusst NICHT extrapoliert, sondern
        # der 32-Hz-Randwert konstant fortgesetzt (siehe Docstring von
        # perceptual_threshold_s()).
        assert perceptual_threshold_s(20.0) == pytest.approx(0.90, abs=1e-9)
        assert perceptual_threshold_s(5.0) == pytest.approx(0.90, abs=1e-9)

    def test_clamped_above_highest_support_point_not_extrapolated(self):
        assert perceptual_threshold_s(200.0) == pytest.approx(0.17, abs=1e-9)
        assert perceptual_threshold_s(500.0) == pytest.approx(0.17, abs=1e-9)


class TestEvaluateDecayAgainstThreshold:
    def test_nan_decay_gives_none_flags(self):
        result = evaluate_decay_against_threshold(float("nan"), 60.0)
        assert result.exceeds_perceptual_threshold is None
        assert result.threshold_s is None

    def test_short_decay_does_not_exceed_threshold(self):
        result = evaluate_decay_against_threshold(0.01, 60.0)
        assert result.exceeds_perceptual_threshold is False

    def test_long_decay_exceeds_threshold(self):
        result = evaluate_decay_against_threshold(5.0, 60.0)
        assert result.exceeds_perceptual_threshold is True

    def test_threshold_crossing_at_exactly_calibrated_support_point(self):
        # 100 Hz ist ein exakter Stuetzpunkt (Schwelle 0.27s, siehe
        # TestPerceptualThreshold) - prueft den Uebergang hoerbar/unhoerbar
        # direkt an der kalibrierten Grenze, nicht nur grob "kurz"/"lang".
        just_below = evaluate_decay_against_threshold(0.26, 100.0)
        exactly_at = evaluate_decay_against_threshold(0.27, 100.0)
        just_above = evaluate_decay_against_threshold(0.28, 100.0)
        assert just_below.exceeds_perceptual_threshold is False
        assert exactly_at.exceeds_perceptual_threshold is False  # strikt ">", Gleichheit zaehlt nicht als Ueberschreitung
        assert just_above.exceeds_perceptual_threshold is True
        assert exactly_at.threshold_s == pytest.approx(0.27, abs=1e-9)
