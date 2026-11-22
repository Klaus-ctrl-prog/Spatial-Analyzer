"""Psychoakustische Wohnzimmer-Zielkurve ("Hauskurve").

Siehe docs/SPECIFICATION.md Abschnitt 3.F und die Projekt-Grundregeln
(raum_akustik.md): sanfter Bassanstieg unter 150 Hz (Hauskurve-Wunsch, kein
Dogma), leichter Hoehenabfall ab 2 kHz, Mittelband dazwischen flach (0 dB
Referenz). Die Kurve ist IMMER ein Modellwert (FieldMeaning.MODEL), nie ein
Messwert - sie ist der Massstab, an dem Messwerte bewertet werden, nicht
selbst eine Messung.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatial_analyzer.models import FieldMeaning

DEFAULT_BASS_BOOST_DB = 4.0
DEFAULT_BASS_BOOST_BELOW_HZ = 150.0
DEFAULT_BASS_FLOOR_HZ = 20.0
DEFAULT_TREBLE_TILT_DB_PER_DECADE = -1.5
DEFAULT_TREBLE_TILT_ABOVE_HZ = 2000.0


@dataclass(frozen=True)
class TargetCurveConfig:
    """Alle Werte sind bewusste Modellannahmen, siehe Modul-Docstring."""

    bass_boost_db: float = DEFAULT_BASS_BOOST_DB
    bass_boost_below_hz: float = DEFAULT_BASS_BOOST_BELOW_HZ
    bass_floor_hz: float = DEFAULT_BASS_FLOOR_HZ
    treble_tilt_db_per_decade: float = DEFAULT_TREBLE_TILT_DB_PER_DECADE
    treble_tilt_above_hz: float = DEFAULT_TREBLE_TILT_ABOVE_HZ
    meaning: FieldMeaning = FieldMeaning.MODEL

    def __post_init__(self) -> None:
        if not 0 < self.bass_floor_hz < self.bass_boost_below_hz:
            raise ValueError("es muss 0 < bass_floor_hz < bass_boost_below_hz gelten")
        if self.bass_boost_below_hz >= self.treble_tilt_above_hz:
            raise ValueError("bass_boost_below_hz muss < treble_tilt_above_hz sein")
        if self.bass_boost_db < 0:
            raise ValueError("bass_boost_db sollte >= 0 sein (Anstieg, kein Abfall)")


def house_curve_db(freq_hz: np.ndarray, config: TargetCurveConfig = TargetCurveConfig()) -> np.ndarray:
    """Relative Zielkurve in dB (0 dB = flaches Mittelband).

    - Unterhalb config.bass_boost_below_hz: linearer Anstieg im log10(f)-
      Raum von 0 dB (bei bass_boost_below_hz) auf +bass_boost_db (bei
      bass_floor_hz); unterhalb bass_floor_hz bleibt der Wert konstant auf
      +bass_boost_db (kein weiterer Anstieg ins Unendliche).
    - Mittelband [bass_boost_below_hz, treble_tilt_above_hz]: 0 dB.
    - Oberhalb config.treble_tilt_above_hz: linearer Abfall im log10(f)-Raum
      mit config.treble_tilt_db_per_decade dB pro Dekade.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    result = np.zeros_like(freq_hz)

    bass_mask = freq_hz < config.bass_boost_below_hz
    if np.any(bass_mask):
        clamped = np.clip(freq_hz[bass_mask], config.bass_floor_hz, config.bass_boost_below_hz)
        log_low = np.log10(config.bass_floor_hz)
        log_high = np.log10(config.bass_boost_below_hz)
        t = (np.log10(clamped) - log_high) / (log_low - log_high)  # 0 bei high, 1 bei low
        result[bass_mask] = t * config.bass_boost_db

    treble_mask = freq_hz > config.treble_tilt_above_hz
    if np.any(treble_mask):
        decades = np.log10(freq_hz[treble_mask] / config.treble_tilt_above_hz)
        result[treble_mask] = decades * config.treble_tilt_db_per_decade

    return result


def absolute_target_db(
    freq_hz: np.ndarray, reference_level_db: float, config: TargetCurveConfig = TargetCurveConfig()
) -> np.ndarray:
    """Zielkurve auf einen absoluten Pegel bezogen (z. B. den energetisch
    gemittelten Referenzpegel einer Messung im Mittelband)."""
    return reference_level_db + house_curve_db(freq_hz, config)
