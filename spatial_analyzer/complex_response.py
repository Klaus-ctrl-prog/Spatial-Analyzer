"""Umrechnung zwischen SPL/Phase-Darstellung und komplexer Uebertragungsfunktion.

Gemeinsame Grundlage fuer die L+R-Summenauswertung (lr_sum.py), die
Subwoofer-Delay-Optimierung (sub_optimizer.py) und die Biquad-
Mehrpunktvorschau (biquad_preview.py): alle drei brauchen phasenrichtige
komplexe Werte, nicht nur Betragskurven in dB.
"""

from __future__ import annotations

import numpy as np


def to_complex(spl_db: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    """SPL (dB) + Phase (Grad) -> komplexe Uebertragungsfunktion.

    Amplitude (linear) = 10**(spl_db/20), Phase in Radiant.

    Haertungshinweis (ab 2026-11-22 verifiziert, siehe test_complex_response.py
    `TestAbsoluteLevelPreservation`): diese Umrechnung ist eine reine,
    punktweise Formel ohne jede Normalisierung (kein Abzug eines Mittelwerts,
    kein Bezug auf einen Referenzpegel). Die absolute SPL-Pegelbasis einer
    Messung bleibt dadurch zwangslaeufig erhalten - jede Verschiebung des
    Pegels muesste an anderer Stelle in der Rechenkette passieren (z. B.
    durch eine bewusste Referenzierung wie in pipeline.py), nicht hier.
    """
    spl_db = np.asarray(spl_db, dtype=np.float64)
    phase_deg = np.asarray(phase_deg, dtype=np.float64)
    if spl_db.shape != phase_deg.shape:
        raise ValueError("spl_db und phase_deg muessen gleich lang sein")
    amplitude = 10.0 ** (spl_db / 20.0)
    phase_rad = np.deg2rad(phase_deg)
    return amplitude * np.exp(1j * phase_rad)


def from_complex(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Komplexe Uebertragungsfunktion -> (spl_db, phase_deg)."""
    values = np.asarray(values, dtype=np.complex128)
    amplitude = np.abs(values)
    with np.errstate(divide="ignore"):
        spl_db = 20.0 * np.log10(np.maximum(amplitude, 1e-12))
    phase_deg = np.rad2deg(np.angle(values))
    return spl_db, phase_deg
