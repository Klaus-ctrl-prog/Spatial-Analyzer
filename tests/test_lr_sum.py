import numpy as np
import pytest

from spatial_analyzer.lr_sum import compare_predicted_vs_measured, predict_complex_sum


class TestPredictComplexSum:
    def test_in_phase_equal_sources_double_amplitude(self):
        spl = np.array([70.0, 70.0])
        phase = np.array([0.0, 0.0])
        result = predict_complex_sum(spl, phase, spl, phase)
        # Zwei kohaerente, gleich starke, gleichphasige Quellen verdoppeln
        # die Amplitude -> +6.02 dB.
        amplitude_db = 20 * np.log10(np.abs(result))
        np.testing.assert_allclose(amplitude_db, 70.0 + 6.0206, atol=0.01)

    def test_out_of_phase_equal_sources_cancel(self):
        spl = np.array([70.0])
        result = predict_complex_sum(spl, np.array([0.0]), spl, np.array([180.0]))
        np.testing.assert_allclose(np.abs(result), [0.0], atol=1e-9)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            predict_complex_sum(
                np.array([70.0, 70.0]), np.array([0.0, 0.0]), np.array([70.0]), np.array([0.0])
            )


class TestComparePredictedVsMeasured:
    def test_perfect_match_gives_zero_rmse(self):
        freq = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
        spl = np.array([60.0, 62.0, 65.0, 63.0, 61.0])
        phase = np.zeros_like(spl)
        predicted = predict_complex_sum(spl, phase, np.zeros_like(spl) - 200, phase)
        # L sehr leise, R = spl mit Phase 0 -> Summe ~= spl selbst (R dominiert)
        measured = spl
        result = compare_predicted_vs_measured(freq, predicted, measured, band_hz=(20.0, 100.0))
        assert result.rmse_db < 0.1

    def test_detects_mismatch(self):
        freq = np.array([20.0, 40.0, 60.0, 80.0])
        # 20*log10(10) = 20 dB; measured bewusst abweichend (15 dB) gewaehlt.
        predicted_complex = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.complex128)
        measured = np.array([15.0, 15.0, 15.0, 15.0])
        result = compare_predicted_vs_measured(freq, predicted_complex, measured, band_hz=(20.0, 80.0))
        assert result.rmse_db > 0.0

    def test_raises_if_band_outside_freq(self):
        freq = np.array([20.0, 40.0])
        predicted = np.array([1.0, 1.0], dtype=np.complex128)
        measured = np.array([0.0, 0.0])
        with pytest.raises(ValueError):
            compare_predicted_vs_measured(freq, predicted, measured, band_hz=(500.0, 600.0))
