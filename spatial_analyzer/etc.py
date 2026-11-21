"""ETC (Energy-Time Curve) basierte Erstreflexions- & Kammfilter-Trennung.

Siehe docs/SPECIFICATION.md Abschnitt 3.D.4 und Zwischenstand-Doku Abschnitt
10.3: fruehe Reflexionen (2-15 ms) erzeugen Kammfilter-Einbrueche, deren
Frequenzbereich (grob 67-500 Hz) sich mit dem Bereich echter Raummoden
(70-150 Hz) ueberlappt. Die Unterscheidung erfolgt hier - anders als in der
geprueften RoomEQ-Referenz (siehe Abschnitt 5a/7a der Zwischenstand-Doku,
die trotz gegenteiliger Eigenaussage nie tatsaechlich raeumliche Konsistenz
prueft) - ueber eine echte Konsistenzpruefung ueber alle 11 Positionen:
ortsfest (an vielen Positionen bei fast derselben Frequenz) -> eher Mode;
nur an wenigen/wandernden Positionen -> eher Kammfilter.

Der Reflexions-Schwellenwert (threshold_db) ist eine Modellannahme/
Platzhalter (siehe Spezifikation G.2 - Kalibrierung anhand realer
Wohnzimmer-Messdaten ist ein offener Punkt, nicht Teil dieses Moduls).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert

DEFAULT_REFLECTION_WINDOW_MS = (2.0, 15.0)
DEFAULT_REFLECTION_THRESHOLD_DB = -20.0  # Platzhalter, siehe Modul-Docstring
DEFAULT_FREQ_TOLERANCE_HZ = 2.0
DEFAULT_CONSISTENCY_THRESHOLD = 0.7  # Anteil der Positionen fuer "ortsfest"


def compute_etc_db(ir_samples: np.ndarray, peak_index: int | None = None) -> np.ndarray:
    """Energy-Time-Curve in dB, auf den Peak der Hilbert-Envelope normiert.

    Verwendet die Betragshuellkurve des analytischen Signals (Hilbert-
    Transformierte), nicht den rohen IR-Betrag - Standardvorgehen fuer ETC.
    """
    ir_samples = np.asarray(ir_samples, dtype=np.float64)
    envelope = np.abs(hilbert(ir_samples))
    if peak_index is None:
        peak_index = int(np.argmax(envelope))
    peak_value = envelope[peak_index]
    if peak_value <= 0:
        raise ValueError("Peak-Envelope-Wert ist <= 0, ETC nicht definiert")
    with np.errstate(divide="ignore"):
        etc_db = 20.0 * np.log10(np.maximum(envelope, 1e-12) / peak_value)
    return etc_db


@dataclass
class ReflectionCandidate:
    delay_ms: float
    level_db: float
    notch_spacing_hz: float


def detect_early_reflections(
    etc_db: np.ndarray,
    sample_interval_s: float,
    peak_index: int,
    window_ms: tuple[float, float] = DEFAULT_REFLECTION_WINDOW_MS,
    threshold_db: float = DEFAULT_REFLECTION_THRESHOLD_DB,
) -> list[ReflectionCandidate]:
    """Findet lokale ETC-Maxima im Fenster [window_ms] relativ zum
    Direktschall-Peak, die ueber threshold_db liegen (relativ zum Peak = 0 dB).
    """
    sample_interval_ms = sample_interval_s * 1000.0
    start_idx = peak_index + int(np.floor(window_ms[0] / sample_interval_ms))
    end_idx = peak_index + int(np.ceil(window_ms[1] / sample_interval_ms))
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, etc_db.size)
    if start_idx >= end_idx:
        return []

    segment = etc_db[start_idx:end_idx]
    candidates: list[ReflectionCandidate] = []
    for i in range(1, segment.size - 1):
        if segment[i] <= threshold_db:
            continue
        if segment[i] >= segment[i - 1] and segment[i] >= segment[i + 1]:
            sample_idx = start_idx + i
            delay_ms = (sample_idx - peak_index) * sample_interval_ms
            if delay_ms <= 0:
                continue
            notch_spacing_hz = 1000.0 / delay_ms
            candidates.append(
                ReflectionCandidate(
                    delay_ms=float(delay_ms),
                    level_db=float(segment[i]),
                    notch_spacing_hz=float(notch_spacing_hz),
                )
            )
    return candidates


def classify_bass_dip(
    freq_hz: float,
    position_freqs_hz: list[float],
    theoretical_mode_candidates_hz: list[float],
    freq_tolerance_hz: float = DEFAULT_FREQ_TOLERANCE_HZ,
    consistency_threshold: float = DEFAULT_CONSISTENCY_THRESHOLD,
) -> str:
    """Klassifiziert einen bei einer Referenzposition gefundenen Einbruch
    (freq_hz) anhand echter raeumlicher Konsistenz ueber alle Positionen.

    `position_freqs_hz`: die (ggf. leicht unterschiedliche) Einbruchsfrequenz
    je Position, nur fuer Positionen, an denen ueberhaupt ein Einbruch in der
    Naehe gefunden wurde (Liste kann kuerzer als 11 sein).
    `theoretical_mode_candidates_hz`: Modellwerte aus RoomModel (siehe
    models.py) - nur zur Einordnung, nicht als Beweis.

    Rueckgabe: "mode_candidate", "comb_filter_candidate" oder "unclassified".
    Dies ist eine Modellklassifikation, kein gesicherter physikalischer
    Nachweis (siehe docs/SPECIFICATION.md Abschnitt 2, "Physikalische
    Ehrlichkeit").
    """
    if not position_freqs_hz:
        return "unclassified"

    close_enough = [abs(f - freq_hz) <= freq_tolerance_hz for f in position_freqs_hz]
    presence_fraction = sum(close_enough) / max(len(position_freqs_hz), 1)

    near_theoretical_mode = any(
        abs(freq_hz - cand) <= freq_tolerance_hz for cand in theoretical_mode_candidates_hz
    )

    if presence_fraction >= consistency_threshold:
        return "mode_candidate" if near_theoretical_mode else "mode_candidate_unmodeled"
    return "comb_filter_candidate"
