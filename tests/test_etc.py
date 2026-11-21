import numpy as np
import pytest

from spatial_analyzer.etc import classify_bass_dip, compute_etc_db, detect_early_reflections


class TestComputeEtcDb:
    def test_peak_is_zero_db(self):
        ir = np.zeros(1000)
        ir[500] = 1.0
        etc_db = compute_etc_db(ir, peak_index=500)
        assert etc_db[500] == pytest.approx(0.0, abs=1e-6)

    def test_values_are_non_positive_at_peak_normalization(self):
        rng = np.random.default_rng(1)
        ir = rng.normal(scale=0.01, size=1000)
        ir[500] += 1.0
        etc_db = compute_etc_db(ir, peak_index=500)
        assert etc_db[500] <= 1e-6

    def test_raises_on_zero_peak(self):
        ir = np.zeros(10)
        with pytest.raises(ValueError):
            compute_etc_db(ir, peak_index=0)


class TestDetectEarlyReflections:
    def test_finds_injected_reflection(self):
        fs = 48000.0
        dt = 1.0 / fs
        n = 2000
        ir = np.zeros(n)
        peak_index = 100
        ir[peak_index] = 1.0
        reflection_delay_ms = 5.0
        reflection_idx = peak_index + int(round(reflection_delay_ms / 1000.0 / dt))
        ir[reflection_idx] = 0.3  # -10.5 dB relativ zum Peak
        etc_db = np.zeros(n)
        etc_db[peak_index] = 0.0
        etc_db[reflection_idx] = 20 * np.log10(0.3)
        etc_db[:] = -80.0
        etc_db[peak_index] = 0.0
        etc_db[reflection_idx] = 20 * np.log10(0.3)

        candidates = detect_early_reflections(
            etc_db, dt, peak_index, window_ms=(2.0, 15.0), threshold_db=-20.0
        )
        assert len(candidates) == 1
        assert candidates[0].delay_ms == pytest.approx(reflection_delay_ms, abs=0.1)
        assert candidates[0].notch_spacing_hz == pytest.approx(1000.0 / reflection_delay_ms, rel=0.02)

    def test_empty_when_window_out_of_range(self):
        etc_db = np.full(10, -80.0)
        result = detect_early_reflections(etc_db, 1e-4, peak_index=5, window_ms=(2.0, 15.0))
        assert result == []

    def test_below_threshold_not_detected(self):
        etc_db = np.full(1000, -80.0)
        etc_db[100] = -40.0
        result = detect_early_reflections(etc_db, 1e-5, peak_index=0, threshold_db=-20.0)
        assert result == []


class TestClassifyBassDip:
    def test_consistent_across_positions_is_mode_candidate(self):
        result = classify_bass_dip(
            freq_hz=60.0,
            position_freqs_hz=[60.0, 60.5, 59.5, 60.2, 60.0, 59.8, 60.1, 60.0, 59.9, 60.3, 60.0],
            theoretical_mode_candidates_hz=[30.0, 46.1, 60.0, 71.5],
        )
        assert result == "mode_candidate"

    def test_inconsistent_across_positions_is_comb_filter(self):
        result = classify_bass_dip(
            freq_hz=120.0,
            position_freqs_hz=[120.0, 150.0],  # nur 2 von z.B. 11 nahe genug
            theoretical_mode_candidates_hz=[30.0, 46.1, 60.0, 71.5],
        )
        # mit nur dieser kleinen Liste: presence_fraction = 1/2 = 0.5 < 0.7
        assert result == "comb_filter_candidate"

    def test_empty_position_list_is_unclassified(self):
        assert classify_bass_dip(60.0, [], [60.0]) == "unclassified"

    def test_consistent_but_not_near_theoretical_mode(self):
        result = classify_bass_dip(
            freq_hz=90.0,
            position_freqs_hz=[90.0] * 10,
            theoretical_mode_candidates_hz=[30.0, 46.1, 60.0, 71.5],
        )
        assert result == "mode_candidate_unmodeled"
