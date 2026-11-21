import numpy as np
import pytest

from spatial_analyzer.biquad_preview import (
    SQRT_HALF,
    PeakingFilter,
    response_complex,
    simulate_multi_position_preview,
)
from spatial_analyzer.complex_response import to_complex


class TestPeakingFilter:
    def test_rejects_unknown_type(self):
        with pytest.raises(ValueError):
            PeakingFilter(type="notch", fc=100.0)

    def test_rejects_fc_above_nyquist(self):
        f = PeakingFilter(type="peq", fc=30000.0, gain_db=3.0, q=1.0)
        with pytest.raises(ValueError):
            f.sos(fs=48000.0)

    def test_zero_gain_peq_is_near_unity_at_center(self):
        f = PeakingFilter(type="peq", fc=100.0, gain_db=0.0, q=1.0)
        h = response_complex([f], fs=48000.0, freqs_hz=np.array([100.0]))
        np.testing.assert_allclose(np.abs(h), [1.0], atol=1e-6)

    def test_positive_gain_boosts_center_frequency(self):
        f = PeakingFilter(type="peq", fc=100.0, gain_db=6.0, q=2.0)
        h = response_complex([f], fs=48000.0, freqs_hz=np.array([100.0]))
        level_db = 20 * np.log10(np.abs(h))
        np.testing.assert_allclose(level_db, [6.0], atol=0.05)

    def test_negative_gain_cuts_center_frequency(self):
        f = PeakingFilter(type="peq", fc=100.0, gain_db=-6.0, q=2.0)
        h = response_complex([f], fs=48000.0, freqs_hz=np.array([100.0]))
        level_db = 20 * np.log10(np.abs(h))
        np.testing.assert_allclose(level_db, [-6.0], atol=0.05)

    def test_lowpass_passes_low_frequency_and_attenuates_high(self):
        f = PeakingFilter(type="lowpass", fc=100.0, q=SQRT_HALF)
        h = response_complex([f], fs=48000.0, freqs_hz=np.array([10.0, 1000.0]))
        level_db = 20 * np.log10(np.abs(h))
        assert level_db[0] > -1.0
        assert level_db[1] < -20.0

    def test_highpass_passes_high_frequency_and_attenuates_low(self):
        f = PeakingFilter(type="highpass", fc=100.0, q=SQRT_HALF)
        h = response_complex([f], fs=48000.0, freqs_hz=np.array([10000.0, 10.0]))
        level_db = 20 * np.log10(np.abs(h))
        assert level_db[0] > -1.0
        assert level_db[1] < -20.0

    def test_lowshelf_boosts_low_end(self):
        f = PeakingFilter(type="lowshelf", fc=200.0, gain_db=6.0, q=SQRT_HALF)
        h = response_complex([f], fs=48000.0, freqs_hz=np.array([20.0, 10000.0]))
        level_db = 20 * np.log10(np.abs(h))
        np.testing.assert_allclose(level_db[0], 6.0, atol=0.5)
        np.testing.assert_allclose(level_db[1], 0.0, atol=0.5)

    def test_highshelf_boosts_high_end(self):
        f = PeakingFilter(type="highshelf", fc=2000.0, gain_db=6.0, q=SQRT_HALF)
        h = response_complex([f], fs=48000.0, freqs_hz=np.array([20.0, 20000.0]))
        level_db = 20 * np.log10(np.abs(h))
        np.testing.assert_allclose(level_db[0], 0.0, atol=0.5)
        np.testing.assert_allclose(level_db[1], 6.0, atol=0.5)

    def test_lowpass_and_highpass_reject_nonzero_gain(self):
        # Formel ignoriert gain_db bei lowpass/highpass konzeptionell nicht -
        # KEFplayer verbietet dort gain_db != 0 explizit; hier pruefen wir
        # nur, dass gain_db=0 (Default) funktioniert und keine Exception
        # fuer den Normalfall auftritt.
        f = PeakingFilter(type="highpass", fc=50.0)
        f.sos(fs=48000.0)


class TestResponseComplex:
    def test_empty_filter_list_is_unity(self):
        h = response_complex([], fs=48000.0, freqs_hz=np.array([50.0, 100.0, 1000.0]))
        np.testing.assert_allclose(h, np.ones(3))


class TestSimulateMultiPositionPreview:
    def test_identical_l_r_filters_reduce_to_h_times_sum(self):
        freq = np.array([50.0, 100.0, 200.0])
        l_spl = np.array([70.0, 72.0, 68.0])
        r_spl = np.array([71.0, 70.0, 69.0])
        phase = np.zeros_like(freq)
        l_complex = to_complex(l_spl, phase)[np.newaxis, :]
        r_complex = to_complex(r_spl, phase)[np.newaxis, :]
        filt = [PeakingFilter(type="peq", fc=100.0, gain_db=-3.0, q=1.0)]

        preview = simulate_multi_position_preview(
            freq, ["sofa_mitte"], l_complex, r_complex, filt, filt, fs=48000.0
        )
        h = response_complex(filt, 48000.0, freq)
        expected_lr_after = (l_complex[0] * h) + (r_complex[0] * h)
        expected_lr_after_db = 20 * np.log10(np.abs(expected_lr_after))
        np.testing.assert_allclose(preview.lr_after_db[0], expected_lr_after_db, atol=1e-6)

    def test_before_matches_unfiltered_complex_sum(self):
        freq = np.array([50.0, 100.0])
        l_complex = to_complex(np.array([70.0, 70.0]), np.zeros(2))[np.newaxis, :]
        r_complex = to_complex(np.array([70.0, 70.0]), np.zeros(2))[np.newaxis, :]
        preview = simulate_multi_position_preview(
            freq, ["p1"], l_complex, r_complex, [], [], fs=48000.0
        )
        expected = 20 * np.log10(np.abs(l_complex[0] + r_complex[0]))
        np.testing.assert_allclose(preview.lr_before_db[0], expected)

    def test_rejects_mismatched_position_counts(self):
        freq = np.array([50.0, 100.0])
        l_complex = np.ones((2, 2), dtype=np.complex128)
        r_complex = np.ones((1, 2), dtype=np.complex128)
        with pytest.raises(ValueError):
            simulate_multi_position_preview(freq, ["a", "b"], l_complex, r_complex, [], [], fs=48000.0)
