import numpy as np
import pytest

from spatial_analyzer.grid import interpolate_to_grid, r160_grid


class TestR160Grid:
    def test_monotonic_increasing(self):
        grid = r160_grid(20.0, 200.0)
        assert np.all(np.diff(grid) > 0)

    def test_within_bounds(self):
        grid = r160_grid(20.0, 200.0)
        assert grid[0] >= 20.0
        assert grid[-1] <= 200.0

    def test_step_ratio_is_r160(self):
        grid = r160_grid(100.0, 200.0)
        ratios = grid[1:] / grid[:-1]
        expected_ratio = 10.0 ** (1.0 / 160.0)
        np.testing.assert_allclose(ratios, expected_ratio, rtol=1e-9)

    def test_rejects_non_positive_bounds(self):
        with pytest.raises(ValueError):
            r160_grid(0.0, 100.0)

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError):
            r160_grid(200.0, 20.0)

    def test_raises_when_no_gridpoint_in_tiny_range(self):
        # Zwischen 100.001 Hz und 101.0 Hz liegt kein R160-Gitterpunkt: der
        # naechste Punkt nach 100.0 Hz liegt bei ca. 101.445 Hz
        # (Verhaeltnisfaktor 10**(1/160)).
        with pytest.raises(ValueError):
            r160_grid(100.001, 101.0)

    @pytest.mark.parametrize(
        "start_hz,end_hz",
        [
            (float("nan"), 100.0),
            (20.0, float("nan")),
            (float("inf"), 100.0),
            (20.0, float("inf")),
        ],
    )
    def test_rejects_nan_and_inf_bounds_with_clear_error(self, start_hz, end_hz):
        # Haertung (Prioritaet 5): vor dem Fix schlug nan als start_hz/end_hz
        # NICHT am "<= 0"-Check an (float('nan') <= 0 ist False) und
        # brach erst spaeter mit einem unklaren rohen
        # "cannot convert float NaN to integer" ab.
        with pytest.raises(ValueError):
            r160_grid(start_hz, end_hz)


class TestR160GridMalformedInputsDoNotCrashRaw:
    """Stellt sicher, dass r160_grid() bei ungueltigen Eingaben konsistent
    ValueError wirft statt mit einer anderen, unerwarteten Exception-Klasse
    abzubrechen (z. B. TypeError aus numpy-internen Operationen)."""

    def test_non_numeric_start_raises_value_or_type_error_not_silently(self):
        with pytest.raises((ValueError, TypeError)):
            r160_grid("abc", 100.0)  # type: ignore[arg-type]


class TestInterpolateToGrid:
    def test_exact_match_at_source_points(self):
        source_freqs = np.array([20.0, 40.0, 80.0, 160.0])
        source_values = np.array([70.0, 72.0, 68.0, 65.0])
        result = interpolate_to_grid(source_freqs, source_values, source_freqs)
        np.testing.assert_allclose(result, source_values)

    def test_no_extrapolation_outside_source_range(self):
        source_freqs = np.array([40.0, 80.0])
        source_values = np.array([70.0, 68.0])
        target = np.array([20.0, 60.0, 160.0])
        result = interpolate_to_grid(source_freqs, source_values, target)
        assert np.isnan(result[0])
        assert not np.isnan(result[1])
        assert np.isnan(result[2])

    def test_respects_valid_range_even_within_source_span(self):
        source_freqs = np.array([20.0, 40.0, 80.0, 160.0])
        source_values = np.array([70.0, 72.0, 68.0, 65.0])
        target = np.array([30.0, 60.0, 100.0])
        result = interpolate_to_grid(
            source_freqs, source_values, target, valid_start_hz=50.0, valid_end_hz=120.0
        )
        assert np.isnan(result[0])  # 30 Hz ausserhalb valid range
        assert not np.isnan(result[1])
        assert not np.isnan(result[2])

    def test_rejects_non_monotonic_source(self):
        with pytest.raises(ValueError):
            interpolate_to_grid(
                np.array([40.0, 20.0]), np.array([1.0, 2.0]), np.array([30.0])
            )

    def test_rejects_duplicate_source_frequencies(self):
        # Doppelte Frequenz = Steigung 0 = ebenfalls kein "streng monoton
        # steigend", muss denselben Fehler wie eine echte Inversion werfen.
        with pytest.raises(ValueError):
            interpolate_to_grid(
                np.array([20.0, 20.0, 100.0]), np.array([1.0, 2.0, 3.0]), np.array([30.0])
            )

    def test_rejects_empty_source_arrays(self):
        # Vorher: numpy's eigener, unklarer Fehler "array of sample points
        # is empty" statt einer verstaendlichen Validierungsmeldung.
        with pytest.raises(ValueError):
            interpolate_to_grid(np.array([]), np.array([]), np.array([30.0]))

    @pytest.mark.parametrize(
        "bad_source_freqs",
        [
            [20.0, float("nan"), 1000.0],
            [20.0, float("inf"), 1000.0],
        ],
    )
    def test_rejects_non_finite_source_freqs(self, bad_source_freqs):
        # Haertung (Prioritaet 5): NaN/Inf im Quellraster schlug vorher
        # NICHT bei der Monotonie-Pruefung an (np.diff([...nan...]) <= 0
        # ist ueberall False) und floss stillschweigend in np.interp() ein -
        # live reproduziert: Ergebnis enthielt NaN an Stellen, die eigentlich
        # gueltige Interpolationspunkte gewesen waeren.
        with pytest.raises(ValueError):
            interpolate_to_grid(
                np.array(bad_source_freqs), np.array([1.0, 2.0, 3.0]), np.array([30.0])
            )

    def test_rejects_non_finite_source_values(self):
        with pytest.raises(ValueError):
            interpolate_to_grid(
                np.array([20.0, 100.0, 1000.0]),
                np.array([1.0, float("nan"), 3.0]),
                np.array([30.0]),
            )

    def test_single_source_point_still_works(self):
        # Grenzfall, der schon vor der Haertung korrekt funktionierte:
        # ein einzelner Quellpunkt ist gueltig (Interpolation an genau
        # diesem Punkt, NaN sonst), soll weiterhin keinen Fehler werfen.
        result = interpolate_to_grid(np.array([100.0]), np.array([60.0]), np.array([100.0, 200.0]))
        assert result[0] == pytest.approx(60.0)
        assert np.isnan(result[1])

    def test_interpolates_linearly_in_log_space(self):
        # Bei gleichem SPL-Wert an beiden Stuetzstellen muss jeder
        # Zwischenwert ebenfalls exakt diesem Wert entsprechen.
        source_freqs = np.array([20.0, 200.0])
        source_values = np.array([70.0, 70.0])
        target = np.array([63.0])
        result = interpolate_to_grid(source_freqs, source_values, target)
        np.testing.assert_allclose(result, [70.0])
