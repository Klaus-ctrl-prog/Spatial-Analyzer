"""Parameterabbildung fuer Mosconi Pico 8/10: Kanalkonfiguration -> normalisierte
Einstelltabelle (Delay auf 0,02-ms-Raster, Abweichung dokumentiert).

Nur Abbildung/Validierung - es wird nichts berechnet und nichts an das Geraet
gesendet; die Tabelle ist zum manuellen Uebertragen gedacht (SMART453_BASS_SPEC.md,
Abschnitt 6).
"""

from __future__ import annotations

from typing import Any

from spatial_analyzer.adapters.mosconi_pico.validation import delay_grid_value, validate_channels


def map_channels(channels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validiert und gibt je Kanal eine Zeile mit Soll-/Rasterwert zurueck.

    Eingabe bleibt unveraendert. Delay: Rasterwert = Schritte * 0,02 ms (Gleichstand: aufrunden, exakte Arithmetik);
    `delay_deviation_ms` = Rasterwert - Sollwert.
    """
    validate_channels(channels)
    rows: list[dict[str, Any]] = []
    for ch in channels:
        want = float(ch.get("delay_ms", 0.0))
        steps, grid, deviation = delay_grid_value(want)
        rows.append({
            "name": ch["name"],
            "role": ch["role"],
            "delay_target_ms": want,
            "delay_steps": steps,
            "delay_ms": grid,
            "delay_deviation_ms": deviation,
            "phase_deg": ch.get("phase_deg", 0),
            "hpf": dict(ch["hpf"]) if ch.get("hpf") else None,
            "lpf": dict(ch["lpf"]) if ch.get("lpf") else None,
            "peq": [dict(f) for f in ch.get("peq", [])],
        })
    return rows
