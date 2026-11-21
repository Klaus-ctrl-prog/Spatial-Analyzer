import numpy as np
import pytest

from spatial_analyzer.complex_response import from_complex, to_complex


class TestToComplex:
    def test_zero_phase_gives_real_positive(self):
        result = to_complex(np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(result, [1.0 + 0.0j])

    def test_180_degree_phase_gives_negative_real(self):
        result = to_complex(np.array([0.0]), np.array([180.0]))
        np.testing.assert_allclose(result, [-1.0 + 0.0j], atol=1e-9)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            to_complex(np.array([0.0, 1.0]), np.array([0.0]))


class TestFromComplex:
    def test_roundtrip_with_to_complex(self):
        spl = np.array([60.0, 70.0, 65.0])
        phase = np.array([0.0, 90.0, -45.0])
        complex_vals = to_complex(spl, phase)
        spl_back, phase_back = from_complex(complex_vals)
        np.testing.assert_allclose(spl_back, spl, atol=1e-9)
        np.testing.assert_allclose(phase_back, phase, atol=1e-9)

    def test_near_zero_amplitude_does_not_raise(self):
        spl_db, phase_deg = from_complex(np.array([1e-20 + 0.0j]))
        assert np.isfinite(spl_db[0])
