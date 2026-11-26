"""Validierung einer Mosconi-Kanalkonfiguration gegen die belegten Geraetegrenzen.

Keine Filterberechnung, Fahrzeuganalyse oder akustische Optimierung; keine
Abhaengigkeit von `smart453_bass/`.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

from spatial_analyzer.adapters.mosconi_pico.capabilities import (
    CHANNEL_ROLES,
    CROSSOVER_SLOPES,
    DELAY_GRID_MS,
    MAX_PEQ_PER_CHANNEL,
    PHASE_OPTIONS_DEG,
    TWEETER_MIN_HPF_HZ,
)


class MosconiValidationError(ValueError):
    pass


def _finite(v: Any) -> bool:
    """math.isfinite ohne OverflowError bei riesigen ganzen Zahlen."""
    try:
        return math.isfinite(v)
    except (OverflowError, TypeError):
        return False


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and _finite(v)


def delay_to_steps(delay_ms: float) -> int:
    """Rasterschritte (0,02 ms) fuer einen Delay-Sollwert.

    Exakte Arithmetik auf dem *dezimalen* Wert (`repr`), damit Gleichstaende
    eindeutig sind und kein Ueberlauf entsteht (beliebig grosse ganze Zahlen):
    Gleichstand (genau zwischen zwei Rasterwerten) -> aufrunden (halb nach
    oben). Beispiele: 0,01 ms -> 1 Schritt (0,02), 0,03 -> 2 (0,04),
    0,05 -> 3 (0,06).
    """
    return _steps_exact(delay_ms)


def _exact(delay_ms: float) -> Fraction:
    # Gleiche Zulaessigkeit wie validate_channel: nur endliche int/float (kein bool, kein str/None);
    # verhindert OverflowError (z. B. 10**400) bzw. uneinheitliche TypeError/ValueError.
    if not _num(delay_ms):
        raise MosconiValidationError(f"delay_ms muss eine endliche Zahl sein (erhalten: {type(delay_ms).__name__}).")
    return Fraction(repr(float(delay_ms)))


def _steps_exact(delay_ms: float) -> int:
    x = _exact(delay_ms) / Fraction(str(DELAY_GRID_MS))  # Schritte als exakter Bruch
    return math.floor(x + Fraction(1, 2))


def delay_grid_value(delay_ms: float) -> tuple[int, float, float]:
    """(Schritte, Rasterwert in ms, Abweichung Rasterwert - Sollwert in ms)."""
    steps = _steps_exact(delay_ms)
    grid = steps * Fraction(str(DELAY_GRID_MS))
    return steps, float(grid), float(grid - _exact(delay_ms))


def validate_channel(ch: Any) -> list[str]:
    """Gibt alle Befunde fuer einen Kanal zurueck (leer = gueltig)."""
    if not isinstance(ch, dict):
        return ["Kanal muss ein Objekt sein."]
    p: list[str] = []
    name = ch.get("name", "?")
    w = f"Kanal {name!r}"
    allowed = {"name", "role", "delay_ms", "phase_deg", "hpf", "lpf", "peq"}
    for k in sorted(set(ch) - allowed):
        p.append(f"{w}: unbekanntes Feld {k!r}.")
    if not (isinstance(ch.get("name"), str) and ch["name"].strip()):
        p.append(f"{w}: name fehlt oder ist leer.")
    if ch.get("role") not in CHANNEL_ROLES:
        p.append(f"{w}: role muss einer von {CHANNEL_ROLES} sein.")
    d = ch.get("delay_ms", 0.0)
    if not _num(d) or d < 0:
        p.append(f"{w}: delay_ms muss eine endliche Zahl >= 0 sein.")
    ph = ch.get("phase_deg", 0)
    if isinstance(ph, bool) or not isinstance(ph, int) or ph not in PHASE_OPTIONS_DEG:
        p.append(f"{w}: phase_deg muss {PHASE_OPTIONS_DEG[0]} oder {PHASE_OPTIONS_DEG[1]} sein.")
    for key in ("hpf", "lpf"):
        x = ch.get(key)
        if x is None:
            continue
        if not isinstance(x, dict) or set(x) != {"frequency_hz", "slope"}:
            p.append(f"{w}: {key} braucht genau frequency_hz und slope.")
            continue
        if not (_num(x["frequency_hz"]) and x["frequency_hz"] > 0):
            p.append(f"{w}: {key}.frequency_hz muss eine Zahl > 0 sein.")
        if x["slope"] not in CROSSOVER_SLOPES:
            p.append(f"{w}: {key}.slope muss einer von {CROSSOVER_SLOPES} sein.")
    if ch.get("role") == "tweeter":
        h = ch.get("hpf")
        if not isinstance(h, dict) or not _num(h.get("frequency_hz")) or h["frequency_hz"] < TWEETER_MIN_HPF_HZ:
            p.append(f"{w}: Hochtoener-Schutz - HPF >= {TWEETER_MIN_HPF_HZ:g} Hz ist Pflicht.")
    peq = ch.get("peq", [])
    if not isinstance(peq, list):
        p.append(f"{w}: peq muss eine Liste sein.")
    else:
        if len(peq) > MAX_PEQ_PER_CHANNEL:
            p.append(f"{w}: {len(peq)} PEQs, maximal {MAX_PEQ_PER_CHANNEL} je Kanal.")
        for i, f in enumerate(peq):
            if not isinstance(f, dict) or set(f) != {"fc_hz", "gain_db", "q"}:
                p.append(f"{w}: peq[{i}] braucht genau fc_hz, gain_db, q.")
            elif not (_num(f["fc_hz"]) and f["fc_hz"] > 0 and _num(f["gain_db"]) and _num(f["q"]) and f["q"] > 0):
                p.append(f"{w}: peq[{i}] hat ungueltige Zahlenwerte.")
    return p


def validate_channels(channels: Any) -> None:
    if not isinstance(channels, list) or not channels:
        raise MosconiValidationError("channels muss eine nicht leere Liste sein.")
    problems: list[str] = []
    names: list[str] = []  # nur String-Namen; andere Typen meldet validate_channel
    for ch in channels:
        problems += validate_channel(ch)
        if isinstance(ch, dict) and isinstance(ch.get("name"), str):
            names.append(ch["name"])
    for n in {n for n in names if names.count(n) > 1}:
        problems.append(f"Kanalname {n!r} mehrfach vergeben.")
    if problems:
        raise MosconiValidationError("Ungueltige Mosconi-Konfiguration:\n- " + "\n- ".join(problems))
