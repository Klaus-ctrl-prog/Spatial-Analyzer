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


class TestAbsoluteLevelPreservation:
    """Haertung (ab 2026-11-22): die absolute SPL-Pegelbasis muss ueber
    to_complex()/from_complex() exakt erhalten bleiben, ohne jede implizite
    Normalisierung (kein Bezug auf einen Mittelwert, keinen Referenzpegel,
    keine Skalierung). Dokumentierte Fehlergrenze fuer dieses reine
    Formel-Paar: 1e-9 dB / 1e-9 Grad (Gleitkomma-Rundung, keine Modellgrenze).
    """

    LEVEL_TOLERANCE_DB = 1e-9
    PHASE_TOLERANCE_DEG = 1e-9

    @pytest.mark.parametrize("spl_db", [-40.0, -3.7, 0.0, 42.7, 70.0, 100.0, 120.0])
    def test_arbitrary_absolute_level_roundtrips_exactly(self, spl_db):
        phase = np.array([-123.4])
        complex_vals = to_complex(np.array([spl_db]), phase)
        spl_back, phase_back = from_complex(complex_vals)
        assert spl_back[0] == pytest.approx(spl_db, abs=self.LEVEL_TOLERANCE_DB)
        assert phase_back[0] == pytest.approx(phase[0], abs=self.PHASE_TOLERANCE_DEG)

    def test_no_implicit_normalization_across_differing_levels(self):
        # Zwei Kanaele mit stark unterschiedlichem, aber jeweils fuer sich
        # bekanntem Pegel duerfen sich nicht gegenseitig beeinflussen - jede
        # Zeile wird unabhaengig und ohne gemeinsame Referenzbildung
        # umgerechnet.
        spl_db = np.array([30.0, 90.0])
        phase_deg = np.array([0.0, 0.0])
        complex_vals = to_complex(spl_db, phase_deg)
        spl_back, _ = from_complex(complex_vals)
        np.testing.assert_allclose(spl_back, spl_db, atol=self.LEVEL_TOLERANCE_DB)
        # Verhaeltnis der linearen Amplituden muss exakt 10**((90-30)/20) sein
        ratio = np.abs(complex_vals[1]) / np.abs(complex_vals[0])
        assert ratio == pytest.approx(10.0 ** ((90.0 - 30.0) / 20.0), rel=1e-12)
