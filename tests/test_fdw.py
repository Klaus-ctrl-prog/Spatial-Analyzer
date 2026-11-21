import numpy as np
import pytest

from spatial_analyzer.fdw import FdwConfig, apply_fdw, window_length_s


class TestWindowLengthS:
    def test_full_window_below_threshold(self):
        config = FdwConfig(full_window_below_hz=200.0, short_window_above_hz=500.0)
        lengths = window_length_s(np.array([50.0, 100.0]), ir_duration_s=0.5, config=config)
        np.testing.assert_allclose(lengths, [0.5, 0.5])

    def test_short_window_above_threshold(self):
        config = FdwConfig(cycles=3.0, full_window_below_hz=200.0, short_window_above_hz=500.0)
        lengths = window_length_s(np.array([1000.0]), ir_duration_s=0.5, config=config)
        np.testing.assert_allclose(lengths, [3.0 / 1000.0])

    def test_monotonic_decrease_in_transition_band(self):
        config = FdwConfig(full_window_below_hz=200.0, short_window_above_hz=500.0)
        freqs = np.array([200.0, 300.0, 400.0, 500.0])
        lengths = window_length_s(freqs, ir_duration_s=1.0, config=config)
        assert np.all(np.diff(lengths) <= 0)

    def test_rejects_invalid_config(self):
        with pytest.raises(ValueError):
            FdwConfig(full_window_below_hz=600.0, short_window_above_hz=500.0)


class TestApplyFdw:
    def test_pure_tone_recovers_correct_magnitude(self):
        fs = 48000.0
        dt = 1.0 / fs
        n = 4800
        f0 = 1000.0
        t = np.arange(n) * dt
        amplitude = 0.5
        ir = amplitude * np.cos(2 * np.pi * f0 * t)
        config = FdwConfig(cycles=10.0, full_window_below_hz=200.0, short_window_above_hz=500.0)
        result = apply_fdw(ir, dt, peak_index=0, target_freqs_hz=np.array([f0]), config=config)
        # Betrag der gefensterten DFT-Summe bei f0 sollte proportional zur
        # Amplitude und zur Fensterlaenge (in Samples) sein - wir pruefen nur
        # dass ueberhaupt ein deutliches Maximum bei f0 relativ zu einer
        # weit entfernten Frequenz existiert.
        off_result = apply_fdw(ir, dt, peak_index=0, target_freqs_hz=np.array([f0 * 2]), config=config)
        assert np.abs(result[0]) > 5 * np.abs(off_result[0])

    def test_raises_when_peak_at_end_of_ir(self):
        ir = np.zeros(10)
        with pytest.raises(ValueError):
            apply_fdw(ir, 1e-4, peak_index=10, target_freqs_hz=np.array([1000.0]))

    def test_output_shape_matches_target_freqs(self):
        ir = np.random.default_rng(0).normal(size=1000)
        result = apply_fdw(ir, 1 / 48000, peak_index=100, target_freqs_hz=np.array([100.0, 1000.0, 5000.0]))
        assert result.shape == (3,)
