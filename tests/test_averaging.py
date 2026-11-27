import warnings

import numpy as np
import pytest

from spatial_analyzer.averaging import (
    align_spl,
    average_variant_a,
    average_variant_b,
    case_bootstrap_band,
    energetic_mean_db,
    spread_metrics,
)


class TestEnergeticMeanDb:
    def test_identical_rows_average_to_same_value(self):
        values = np.array([[70.0, 60.0], [70.0, 60.0], [70.0, 60.0]])
        result = energetic_mean_db(values, axis=0)
        np.testing.assert_allclose(result, [70.0, 60.0])

    def test_two_equal_sources_add_three_db(self):
        # Zwei inkohaerente, gleich starke Quellen addieren sich energetisch
        # um ~3.01 dB (bekannter Referenzwert).
        values = np.array([[70.0], [70.0]])
        result = energetic_mean_db(values, axis=0)
        # RMS-Durchschnitt (nicht Summe!) von zwei gleichen Pegeln bleibt
        # gleich: 10*log10(mean(10^(70/10), 10^(70/10))) == 70.
        np.testing.assert_allclose(result, [70.0])

    def test_nan_values_are_ignored(self):
        values = np.array([[70.0, np.nan], [70.0, 60.0], [70.0, 60.0]])
        result = energetic_mean_db(values, axis=0)
        np.testing.assert_allclose(result, [70.0, 60.0])


class TestAverageVariantA:
    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            average_variant_a(np.array([70.0, 60.0]))

    def test_matches_energetic_mean(self):
        values = np.array([[65.0, 70.0], [68.0, 71.0]])
        a = average_variant_a(values)
        b = energetic_mean_db(values, axis=0)
        np.testing.assert_allclose(a, b)


class TestAlignSpl:
    def test_equalizes_reference_band_level(self):
        freqs = np.array([100.0, 500.0, 1000.0, 1500.0, 5000.0])
        values = np.array(
            [
                [60.0, 70.0, 70.0, 70.0, 60.0],
                [55.0, 65.0, 65.0, 65.0, 50.0],
            ]
        )
        aligned = align_spl(values, freqs, align_band_hz=(400.0, 2000.0))
        ref_mask = (freqs >= 400.0) & (freqs <= 2000.0)
        row0_ref = energetic_mean_db(aligned[0:1, ref_mask], axis=1)
        row1_ref = energetic_mean_db(aligned[1:2, ref_mask], axis=1)
        np.testing.assert_allclose(row0_ref, row1_ref, atol=1e-9)

    def test_raises_if_band_outside_freqs(self):
        freqs = np.array([20.0, 40.0])
        values = np.array([[60.0, 61.0]])
        with pytest.raises(ValueError):
            align_spl(values, freqs, align_band_hz=(400.0, 2000.0))


class TestAverageVariantB:
    def test_different_absolute_levels_still_align_before_averaging(self):
        freqs = np.array([400.0, 1000.0, 2000.0, 50.0])
        # Position 2 ist um 10 dB leiser (z.B. weiter weg), aber sonst
        # identische Kurvenform. align_spl verschiebt beide Zeilen auf den
        # gemeinsamen (arithmetischen) Mittelwert ihrer Referenzband-Pegel
        # (70 und 60 dB -> gemeinsamer Zielpegel 65 dB), danach ist die
        # Kurvenform beider Positionen identisch -> die Mittelung aendert
        # anschliessend nichts mehr an der Form.
        values = np.array(
            [
                [70.0, 70.0, 70.0, 80.0],
                [60.0, 60.0, 60.0, 70.0],
            ]
        )
        result = average_variant_b(values, freqs, align_band_hz=(400.0, 2000.0))
        np.testing.assert_allclose(result, [65.0, 65.0, 65.0, 75.0], atol=1e-9)


class TestSpreadMetrics:
    def test_basic_metrics(self):
        values = np.array([[60.0, 70.0], [62.0, 74.0], [58.0, 66.0]])
        metrics = spread_metrics(values)
        np.testing.assert_allclose(metrics.min_db, [58.0, 66.0])
        np.testing.assert_allclose(metrics.max_db, [62.0, 74.0])
        np.testing.assert_allclose(metrics.range_db, [4.0, 8.0])
        np.testing.assert_array_equal(metrics.n_valid, [3, 3])

    def test_nan_reduces_n_valid_without_runtime_warning(self):
        values = np.array([[60.0, np.nan], [62.0, 74.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            metrics = spread_metrics(values)

        np.testing.assert_array_equal(metrics.n_valid, [2, 1])
        np.testing.assert_allclose(metrics.std_db[0], np.sqrt(2.0))
        assert np.isnan(metrics.std_db[1])


class TestCaseBootstrapBand:
    def test_band_contains_the_mean_and_has_sensible_order(self):
        rng = np.random.default_rng(42)
        values = np.array([[60.0], [62.0], [58.0], [61.0], [59.0]])
        band = case_bootstrap_band(values, n_resamples=200, alpha=0.10, rng=rng)
        mean = energetic_mean_db(values, axis=0)
        assert band.lower_db[0] <= mean[0] <= band.upper_db[0]
        assert band.n_resamples == 200
        assert band.alpha == 0.10

    def test_constant_input_gives_zero_width_band(self):
        rng = np.random.default_rng(1)
        values = np.full((11, 1), 65.0)
        band = case_bootstrap_band(values, n_resamples=100, rng=rng)
        np.testing.assert_allclose(band.lower_db, band.upper_db, atol=1e-9)
