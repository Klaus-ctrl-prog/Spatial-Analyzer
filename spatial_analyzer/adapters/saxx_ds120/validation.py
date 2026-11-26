"""Hardware-Grenzen des Saxx DS 120 (Quelle: AGENTS.md 4.2) und Pruefung des
Builder-kompatiblen Edit-JSON.

Der Adapter beschreibt nur *was* editiert werden darf; die eigentliche
.datdsp-Erzeugung macht der externe `Candidate Builder 0.15.0-qa` (nicht Teil
dieses Repos). `opaque`-Bytes werden nie editiert; `assert_opaque_preserved`
ist die Gegenprobe fuer Template vs. Kandidat.
"""

from __future__ import annotations

import math
from typing import Any

from spatial_analyzer.adapters.saxx_ds120.quantization import SaxxAdapterError

BUILDER_VERSION = "0.15.0-qa"
HARDWARE_VALIDATED = False  # Hardware-Gate: bleibt false bis zum physischen Rundtest (ab April 2027)

EQ_TYPES_BANDS_1_9 = ("PEQ", "LS6", "LS12", "HS6", "HS12")
EQ_TYPES_BANDS_10_14 = ("PEQ",)
EQ_BANDS = 14
XOVER_TYPES = (
    "12db-Bworth", "12db-LR", "12db-Bessel",
    "18db-Bworth", "18db-LR", "18db-Bessel",
    "24db-Bworth", "24db-LR", "24db-Bessel",
    "48db-Bworth", "48db-LR", "48db-Bessel",
)
BYPASS_ACTIVE = 0  # HPF/LPF-Bypass-Flag: 0 = aktiv
BYPASS_BYPASSED = 1  # 1 = Bypass
PHASE_STEPS_DEG = tuple(range(0, 360, 45))
# Opaque / unmapped Bytes der 212-Byte-Datei: unveraendert aus dem Template.
# Bytes 146-169 sind NICHT als Extra-EQs bewiesen und werden nie so behandelt.
OPAQUE_BYTES = frozenset({11, 16, 21, 25, *range(27, 34), *range(146, 170), 210, 211})
PROFILE_SIZE = 212
_ALLOWED_TOP = {"name", "phase_deg", "limiter_db", "output_gain_db", "mute", "bass_boost", "hpf", "lpf", "eq"}
_ALLOWED_EQ = {"band", "frequency_hz", "type", "gain_db", "bypass", "q"}
_ALLOWED_XO = {"frequency_hz", "type", "enabled"}


class ValidationError(SaxxAdapterError):
    pass


def _finite(v: Any) -> bool:
    """math.isfinite ohne OverflowError bei riesigen ganzen Zahlen."""
    try:
        return math.isfinite(v)
    except (OverflowError, TypeError):
        return False


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and _finite(v)


def validate_edit_json(edit: Any, tables: dict[str, tuple[float, ...]] | None = None) -> None:
    """Spiegelt die Regeln des Candidate Builders (unabhaengig implementiert).

    Mit `tables` werden zusaetzlich alle Werte als exakte Tabellenwerte verlangt
    (der Mapper gibt nur Tabellenwerte aus, sodass der Builder nicht erneut
    rundet).
    """
    problems: list[str] = []
    if not isinstance(edit, dict):
        raise ValidationError("Edit-JSON muss ein Objekt sein.")
    for k in sorted(set(edit) - _ALLOWED_TOP):
        problems.append(f"Unbekannter Edit-Schluessel {k!r}.")
    ph = edit.get("phase_deg")
    if "phase_deg" in edit and (isinstance(ph, bool) or not isinstance(ph, int) or ph not in PHASE_STEPS_DEG):
        problems.append(f"phase_deg muss einer von {PHASE_STEPS_DEG} sein (45-Grad-Stufen).")
    name = edit.get("name")
    if "name" in edit:
        if not isinstance(name, str) or not name.isascii() or not 1 <= len(name) <= 19 or not name.isprintable():
            problems.append("name muss 1-19 druckbare ASCII-Zeichen haben.")
    for key in ("hpf", "lpf"):
        if key not in edit:
            continue
        x = edit[key]
        if not isinstance(x, dict):
            problems.append(f"{key} muss ein Objekt sein.")
            continue
        for k in sorted(set(x) - _ALLOWED_XO):
            problems.append(f"{key}: unbekannter Schluessel {k!r}.")
        if "type" in x and x["type"] not in XOVER_TYPES:
            problems.append(f"{key}: Weichentyp {x['type']!r} nicht unterstuetzt.")
        if "enabled" in x and not isinstance(x["enabled"], bool):
            problems.append(f"{key}.enabled muss true/false sein (False = Bypass-Flag 1, True = Flag 0).")
        if "frequency_hz" in x:
            if not _is_num(x["frequency_hz"]):
                problems.append(f"{key}.frequency_hz muss eine endliche Zahl sein.")
            elif tables is not None and x["frequency_hz"] not in tables["frequency_hz"]:
                problems.append(f"{key}.frequency_hz {x['frequency_hz']} ist kein Tabellenwert.")
    eq = edit.get("eq", [])
    if not isinstance(eq, list):
        problems.append("eq muss eine Liste sein.")
        eq = []
    seen: set[int] = set()
    for item in eq:
        if not isinstance(item, dict):
            problems.append("Jedes eq-Element muss ein Objekt sein.")
            continue
        for k in sorted(set(item) - _ALLOWED_EQ):
            problems.append(f"EQ: unbekannter Schluessel {k!r}.")
        b = item.get("band")
        if isinstance(b, bool) or not isinstance(b, int) or not 1 <= b <= EQ_BANDS:
            problems.append("EQ band muss eine ganze Zahl 1-14 sein.")
            continue
        if b in seen:
            problems.append(f"EQ band {b} mehrfach angegeben.")
        seen.add(b)
        allowed = EQ_TYPES_BANDS_1_9 if b <= 9 else EQ_TYPES_BANDS_10_14
        if b >= 10 and ("type" in item or "bypass" in item):
            problems.append(f"EQ {b}: Baender 10-14 haben weder Typ- noch Bypass-Feld (nur PEQ, Felder bleiben im Template).")
        if "type" in item and item["type"] not in allowed:
            problems.append(f"EQ {b}: Typ {item['type']!r} nicht zulaessig (erlaubt: {allowed}).")
        if "bypass" in item and not isinstance(item["bypass"], bool):
            problems.append(f"EQ {b}: bypass muss true/false sein.")
        for field, table in (("frequency_hz", "frequency_hz"), ("gain_db", "gain_db"), ("q", "q")):
            if field in item:
                if not _is_num(item[field]):
                    problems.append(f"EQ {b}: {field} muss eine endliche Zahl sein.")
                elif tables is not None and item[field] not in tables[table]:
                    problems.append(f"EQ {b}: {field} {item[field]} ist kein Tabellenwert.")
    if problems:
        raise ValidationError("Ungueltiges Edit-JSON:\n- " + "\n- ".join(problems))


def assert_opaque_preserved(template: bytes | bytearray, candidate: bytes | bytearray) -> None:
    """Prueft, dass alle opaque Bytes im Kandidaten dem Template entsprechen.

    Reine Bytevergleichs-Hilfe (keine Datei-/Hardware-I/O). Beide Eingaben
    sind die 212 dekodierten Profilbytes.
    """
    if len(template) != PROFILE_SIZE or len(candidate) != PROFILE_SIZE:
        raise ValidationError(f"Profile muessen je {PROFILE_SIZE} Bytes lang sein.")
    changed = sorted(i for i in OPAQUE_BYTES if template[i] != candidate[i])
    if changed:
        raise ValidationError(f"Opaque Bytes veraendert: {changed}")
