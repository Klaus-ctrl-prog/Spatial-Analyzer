import numpy as np
import pytest

from spatial_analyzer.models import FieldMeaning
from spatial_analyzer.target_curve import TargetCurveConfig, absolute_target_db, house_curve_db


class TestTargetCurveConfig:
    def test_default_meaning_is_model(self):
        assert TargetCurveConfig().meaning == FieldMeaning.MODEL

    def test_rejects_inverted_bass_range(self):
        with pytest.raises(ValueError):
            TargetCurveConfig(bass_floor_hz=200.0, bass_boost_below_hz=150.0)

    def test_rejects_overlapping_bands(self):
        with pytest.raises(ValueError):
            TargetCurveConfig(bass_boost_below_hz=3000.0, treble_tilt_above_hz=2000.0)

    def test_rejects_negative_bass_boost(self):
        with pytest.raises(ValueError):
            TargetCurveConfig(bass_boost_db=-2.0)


class TestHouseCurveDb:
    def test_midband_is_flat_zero(self):
        freq = np.array([200.0, 500.0, 1000.0, 1900.0])
        result = house_curve_db(freq)
        np.testing.assert_allclose(result, np.zeros_like(freq))

    def test_bass_boost_at_floor_frequency(self):
        config = TargetCurveConfig(bass_boost_db=4.0, bass_floor_hz=20.0, bass_boost_below_hz=150.0)
        result = house_curve_db(np.array([20.0]), config)
        np.testing.assert_allclose(result, [4.0], atol=1e-9)

    def test_bass_boost_tapers_to_zero_at_threshold(self):
        config = TargetCurveConfig(bass_boost_db=4.0, bass_floor_hz=20.0, bass_boost_below_hz=150.0)
        result = house_curve_db(np.array([150.0]), config)
        np.testing.assert_allclose(result, [0.0], atol=1e-9)

    def test_below_floor_frequency_stays_clamped(self):
        config = TargetCurveConfig(bass_boost_db=4.0, bass_floor_hz=20.0, bass_boost_below_hz=150.0)
        result = house_curve_db(np.array([10.0]), config)
        np.testing.assert_allclose(result, [4.0], atol=1e-9)

    def test_treble_tilt_is_negative_above_threshold(self):
        config = TargetCurveConfig(treble_tilt_above_hz=2000.0, treble_tilt_db_per_decade=-3.0)
        result = house_curve_db(np.array([20000.0]), config)
        # eine Dekade oberhalb 2000 Hz -> genau -3 dB
        np.testing.assert_allclose(result, [-3.0], atol=1e-9)

    def test_monotonic_bass_rise_towards_low_frequencies(self):
        freq = np.array([30.0, 50.0, 100.0, 140.0])
        result = house_curve_db(freq)
        assert np.all(np.diff(result) <= 0)  # steigt zu tiefen Frequenzen hin an


class TestAbsoluteTargetDb:
    def test_adds_reference_level(self):
        freq = np.array([1000.0])
        result = absolute_target_db(freq, reference_level_db=70.0)
        np.testing.assert_allclose(result, [70.0], atol=1e-9)
