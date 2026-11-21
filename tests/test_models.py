import numpy as np
import pytest

from spatial_analyzer.models import (
    Channel,
    FieldMeaning,
    FrequencyGrid,
    ImpulseResponseData,
    Measurement,
    RoomModel,
)


class TestRoomModel:
    def test_axial_mode_frequencies_match_known_values(self):
        # Raummasse aus docs/SPECIFICATION.md: 5.72 x 3.72 x 2.40 m
        room = RoomModel(length_m=5.72, width_m=3.72, height_m=2.40)
        candidates = room.axial_mode_candidates(max_order=2)
        # Modellwerte aus der Zwischenstand-Doku Abschnitt 5 (~30/46/60/71.5 Hz)
        assert candidates["length_1"] == pytest.approx(29.97, abs=0.1)
        assert candidates["width_1"] == pytest.approx(46.10, abs=0.1)
        assert candidates["length_2"] == pytest.approx(59.97, abs=0.1)
        assert candidates["height_1"] == pytest.approx(71.46, abs=0.1)

    def test_rejects_non_positive_dimension(self):
        with pytest.raises(ValueError):
            RoomModel(length_m=0.0, width_m=3.72, height_m=2.40)

    def test_axial_mode_frequency_rejects_order_below_one(self):
        room = RoomModel(length_m=5.72, width_m=3.72, height_m=2.40)
        with pytest.raises(ValueError):
            room.axial_mode_frequency(room.length_m, order=0)


class TestFrequencyGrid:
    def test_defaults_valid_range_to_full_span(self):
        grid = FrequencyGrid(frequencies_hz=np.array([20.0, 50.0, 100.0]))
        assert grid.valid_start_hz == 20.0
        assert grid.valid_end_hz == 100.0
        assert np.all(grid.valid_mask)

    def test_rejects_non_monotonic(self):
        with pytest.raises(ValueError):
            FrequencyGrid(frequencies_hz=np.array([20.0, 20.0, 100.0]))

    def test_rejects_too_short(self):
        with pytest.raises(ValueError):
            FrequencyGrid(frequencies_hz=np.array([20.0]))

    def test_overlap_with_disjoint_ranges_is_none(self):
        a = FrequencyGrid(frequencies_hz=np.array([20.0, 50.0]), valid_start_hz=20, valid_end_hz=50)
        b = FrequencyGrid(frequencies_hz=np.array([60.0, 90.0]), valid_start_hz=60, valid_end_hz=90)
        assert a.overlap_with(b) is None

    def test_overlap_with_partial_overlap(self):
        a = FrequencyGrid(frequencies_hz=np.array([20.0, 80.0]), valid_start_hz=20, valid_end_hz=80)
        b = FrequencyGrid(frequencies_hz=np.array([50.0, 120.0]), valid_start_hz=50, valid_end_hz=120)
        assert a.overlap_with(b) == (50.0, 80.0)


class TestImpulseResponseData:
    def test_peak_index_autodetected(self):
        samples = np.array([0.0, 0.1, 0.9, -0.2, 0.05])
        ir = ImpulseResponseData(samples=samples, sample_interval_s=1 / 48000)
        assert ir.peak_index == 2

    def test_rejects_zero_interval(self):
        with pytest.raises(ValueError):
            ImpulseResponseData(samples=np.array([1.0, 0.0]), sample_interval_s=0.0)

    def test_is_sweep_true_for_multi_sample(self):
        ir = ImpulseResponseData(samples=np.array([0.0, 1.0, 0.0]), sample_interval_s=1e-5)
        assert ir.is_sweep


class TestMeasurement:
    def _make_measurement(self, ppo=None):
        grid = FrequencyGrid(frequencies_hz=np.array([20.0, 40.0, 80.0, 160.0]))
        spl = np.array([70.0, 72.0, 68.0, 65.0])
        return Measurement(
            uuid="abc-123",
            channel=Channel.L,
            position_id="sofa_mitte",
            grid=grid,
            spl_db=spl,
            ppo=ppo,
        )

    def test_rejects_mismatched_spl_length(self):
        grid = FrequencyGrid(frequencies_hz=np.array([20.0, 40.0, 80.0]))
        with pytest.raises(ValueError):
            Measurement(
                uuid="x",
                channel=Channel.L,
                position_id="p1",
                grid=grid,
                spl_db=np.array([1.0, 2.0]),
            )

    def test_96_ppo_sets_forced_smoothing_flag(self):
        m = self._make_measurement(ppo=96)
        assert m.forced_smoothing_applied is True

    def test_below_96_ppo_does_not_force_smoothing(self):
        m = self._make_measurement(ppo=48)
        assert m.forced_smoothing_applied is False

    def test_spl_in_band_respects_valid_range(self):
        m = self._make_measurement()
        band = m.spl_in_band(30.0, 100.0)
        np.testing.assert_allclose(band, [72.0, 68.0])

    def test_has_usable_impulse_response_false_without_ir(self):
        m = self._make_measurement()
        assert m.has_usable_impulse_response is False

    def test_field_meaning_enum_values(self):
        assert FieldMeaning.MODEL.value == "model"
        assert FieldMeaning.CONFIRMED.value == "confirmed"
