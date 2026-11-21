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
    def test_monotonic_decreasing_with_frequency(self):
        values = [perceptual_threshold_s(f) for f in (20.0, 50.0, 100.0, 200.0)]
        assert all(earlier > later for earlier, later in zip(values, values[1:]))

    def test_clamped_outside_model_range(self):
        low = perceptual_threshold_s(5.0)
        at_boundary = perceptual_threshold_s(20.0)
        assert low == pytest.approx(at_boundary)


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
