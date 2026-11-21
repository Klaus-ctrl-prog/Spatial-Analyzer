"""Dritte Pruefebene: L+R-Summenauswertung an der Sofamitte.

Siehe docs/SPECIFICATION.md Abschnitt 3.D.2 und Zwischenstand-Doku
Abschnitt 10.1: primaer die reale akustische L+R-Summenmessung, zusaetzlich
die komplexe (phasenrichtige) Addition aus den Einzelmessungen als
Konsistenz-Check - beide Pfade werden hier unterstuetzt, keiner ersetzt den
anderen.

Interferenzen zwischen L und R (Kammfilterung/Ausloeschung durch
unterschiedliche Pfadlaengen) werden vor allem im Bereich 20-80 Hz erwartet
(siehe Spezifikation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatial_analyzer.complex_response import from_complex, to_complex

DEFAULT_INTERFERENCE_BAND_HZ = (20.0, 80.0)


@dataclass
class LrSumComparison:
    freq_hz: np.ndarray
    predicted_spl_db: np.ndarray
    measured_spl_db: np.ndarray
    diff_db: np.ndarray
    rmse_db: float
    band_hz: tuple[float, float]


def predict_complex_sum(
    l_spl_db: np.ndarray, l_phase_deg: np.ndarray, r_spl_db: np.ndarray, r_phase_deg: np.ndarray
) -> np.ndarray:
    """Phasenrichtige komplexe L+R-Addition aus den getrennten Einzelmessungen.

    Voraussetzung (siehe Zwischenstand-Doku Abschnitt 10.1): L- und
    R-Messung muessen eine gemeinsame Zeitbasis haben (unveraenderte
    Messkette zwischen beiden Messungen).
    """
    l_complex = to_complex(l_spl_db, l_phase_deg)
    r_complex = to_complex(r_spl_db, r_phase_deg)
    if l_complex.shape != r_complex.shape:
        raise ValueError("L- und R-Arrays muessen dieselbe Laenge haben")
    return l_complex + r_complex


def compare_predicted_vs_measured(
    freq_hz: np.ndarray,
    predicted_sum_complex: np.ndarray,
    measured_sum_spl_db: np.ndarray,
    band_hz: tuple[float, float] = DEFAULT_INTERFERENCE_BAND_HZ,
) -> LrSumComparison:
    """Vergleicht die berechnete komplexe Summe mit der real gemessenen
    L+R-Summenkurve im angegebenen Interferenz-Band.

    Das ist ein Konsistenz-Check (siehe Abschnitt 10.1), kein Ersatz fuer
    die reale Messung - eine grosse Abweichung deutet typischerweise auf
    eine verletzte Zeitbasis-Voraussetzung oder auf Nichtlinearitaeten hin,
    nicht automatisch auf einen Messfehler der realen Summenmessung.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    predicted_spl_db, _ = from_complex(predicted_sum_complex)
    measured_spl_db = np.asarray(measured_sum_spl_db, dtype=np.float64)
    if not (freq_hz.shape == predicted_spl_db.shape == measured_spl_db.shape):
        raise ValueError("freq_hz, predicted und measured muessen gleich lang sein")

    band_mask = (freq_hz >= band_hz[0]) & (freq_hz <= band_hz[1])
    if not np.any(band_mask):
        raise ValueError(f"Band {band_hz} Hz liegt ausserhalb von freq_hz")

    diff_db = predicted_spl_db - measured_spl_db
    band_diff = diff_db[band_mask]
    valid = ~np.isnan(band_diff)
    if not np.any(valid):
        rmse_db = float("nan")
    else:
        rmse_db = float(np.sqrt(np.nanmean(band_diff[valid] ** 2)))

    return LrSumComparison(
        freq_hz=freq_hz,
        predicted_spl_db=predicted_spl_db,
        measured_spl_db=measured_spl_db,
        diff_db=diff_db,
        rmse_db=rmse_db,
        band_hz=band_hz,
    )
