import numpy as np
import pytest

from spatial_analyzer.sub_optimizer import ObjectiveWeights, optimize


def _constant_complex(value_db, n_freqs, phase_deg=0.0):
    amp = 10.0 ** (value_db / 20.0)
    return np.full(n_freqs, amp * np.exp(1j * np.deg2rad(phase_deg)), dtype=np.complex128)


class TestObjectiveWeights:
    def test_rejects_weights_not_summing_to_one(self):
        with pytest.raises(ValueError):
            ObjectiveWeights(target_weight=0.5, spatial_weight=0.6)

    def test_default_weights_sum_to_one(self):
        w = ObjectiveWeights()
        assert w.target_weight + w.spatial_weight == pytest.approx(1.0)


class TestOptimize:
    def test_finds_zero_delay_zero_polarity_for_matched_identical_sources(self):
        # H_LR und H_Sub identisch und phasengleich -> beste Loesung sollte
        # 0 Grad Polaritaet und (wegen Periodizitaet) kleines Delay sein,
        # ausser das Optimum liegt bei einer Gain-Anpassung.
        freq = np.linspace(50.0, 120.0, 50)
        n_positions = 3
        h_lr = np.tile(_constant_complex(70.0, freq.size), (n_positions, 1))
        h_sub = np.tile(_constant_complex(70.0, freq.size), (n_positions, 1))
        target = np.full(freq.size, 76.0)  # Ziel: +6dB ueber Einzelquelle (konstruktive Summe)

        result = optimize(
            freq, h_lr, h_sub, target,
            band_hz=(50.0, 120.0),
            gain_range_db=(-6.0, 6.0), gain_step_db=1.0,
            delay_range_ms=(0.0, 2.0), delay_step_ms=0.5,
            polarities_deg=(0.0, 180.0),
        )
        assert result.best_polarity_deg == 0.0
        assert result.best_delay_ms == pytest.approx(0.0, abs=0.01)

    def test_mode_band_rmse_is_finite_when_band_present(self):
        freq = np.linspace(50.0, 120.0, 71)  # 1 Hz Schritte, enthaelt 60-62
        n_positions = 2
        h_lr = np.tile(_constant_complex(70.0, freq.size), (n_positions, 1))
        h_sub = np.tile(_constant_complex(65.0, freq.size), (n_positions, 1))
        target = np.full(freq.size, 70.0)
        result = optimize(
            freq, h_lr, h_sub, target,
            gain_range_db=(-2.0, 2.0), gain_step_db=2.0,
            delay_range_ms=(0.0, 1.0), delay_step_ms=1.0,
            polarities_deg=(0.0, 180.0),
        )
        assert np.isfinite(result.mode_band_rmse_at_best_db)
        assert result.mode_band_hz == (60.0, 62.0)

    def test_rejects_mismatched_position_counts(self):
        freq = np.linspace(50.0, 120.0, 10)
        h_lr = np.tile(_constant_complex(70.0, freq.size), (2, 1))
        h_sub = np.tile(_constant_complex(70.0, freq.size), (3, 1))
        target = np.full(freq.size, 70.0)
        with pytest.raises(ValueError):
            optimize(freq, h_lr, h_sub, target)

    def test_rejects_band_outside_freq_range(self):
        freq = np.linspace(200.0, 300.0, 10)
        h_lr = np.tile(_constant_complex(70.0, freq.size), (1, 1))
        h_sub = np.tile(_constant_complex(70.0, freq.size), (1, 1))
        target = np.full(freq.size, 70.0)
        with pytest.raises(ValueError):
            optimize(freq, h_lr, h_sub, target, band_hz=(50.0, 120.0))
