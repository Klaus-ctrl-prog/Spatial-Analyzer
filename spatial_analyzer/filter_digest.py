"""Kanonischer Digest der Filterlisten (Digest-Domaene `spatial_analyzer.filters.v2`).

Vertragsrahmen: docs/SPECIFICATION.md (filter_export.json); massgeblich fuer
die Pruefregeln ist AGENTS.md 4.1.

Zweck: Integritaetspruefung. Die Do-No-Harm-Pruefung (eq_solver.validate_no_harm)
und der Export (filter_export.py) berechnen denselben Digest ueber die
Filterlisten; stimmen beide nicht ueberein, ist die Freigabe nicht gueltig.
Der Digest ist KEIN Authentizitaetsnachweis (kein Geheimnis, keine Signatur):
er erkennt nachtraegliche Aenderungen, schuetzt aber nicht vor gezielter
Faelschung.

Regeln:
- Erfasst je Filter type, fc_hz, gain_db, q und shelf_slope_db (None, 6 oder 12).
  `reason` ist reine Erlaeuterung und fliesst nicht ein; andere Felder sind
  unbekannt und fuehren zu einem Fehler.
- Zahlen werden ueber canonical_float normiert. -0.0 wird auf 0.0 abgebildet
  (G3): beide Werte haben dieselbe Wirkung und duerfen keinen anderen Digest
  ergeben. NaN, Inf, bool, Strings und Ganzzahlen ausserhalb des float-Bereichs
  werden abgewiesen.
- Reihenfolge innerhalb jeder Liste und die Zugehoerigkeit zu einer der drei
  Listen (eq_proposals_l, eq_proposals_r, subwoofer) fliessen ein.
- Cycle-frei: dieses Modul importiert nur die Standardbibliothek; eq_solver,
  filter_export und die Adapter duerfen es importieren, nicht umgekehrt.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

DIGEST_DOMAIN = "spatial_analyzer.filters.v2"
FILTER_LISTS = ("eq_proposals_l", "eq_proposals_r", "subwoofer")
_REQUIRED_KEYS = ("type", "fc_hz", "gain_db", "q")
_OPTIONAL_KEYS = ("shelf_slope_db", "reason")
_IGNORED_KEYS = ("reason",)


class FilterDigestError(ValueError):
    """Eingabe ist nicht digestierbar (Typ, Wert oder Struktur)."""


def canonical_float(value: Any) -> float:
    """Normiert eine Zahl fuer den Digest; -0.0 wird zu 0.0 (G3)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FilterDigestError(f"Zahl erwartet, erhalten {type(value).__name__}.")
    try:
        v = float(value)
    except OverflowError as exc:  # riesige ganze Zahlen passen nicht in float
        raise FilterDigestError("Zahl ausserhalb des float-Bereichs.") from exc
    if not math.isfinite(v):
        raise FilterDigestError("Nicht-endliche Zahl ist nicht digestierbar.")
    return 0.0 if v == 0.0 else v


def _canonical_filter(f: Mapping) -> list[Any]:
    if not isinstance(f, Mapping):
        raise FilterDigestError("Filter muss ein Objekt sein.")
    for k in _REQUIRED_KEYS:
        if k not in f:
            raise FilterDigestError(f"Pflichtfeld {k!r} fehlt.")
    for k in sorted(set(f) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS)):
        if k in _IGNORED_KEYS:
            continue
        raise FilterDigestError(f"Unbekanntes Feld {k!r}.")
    ftype = f["type"]
    if not isinstance(ftype, str):
        raise FilterDigestError("type muss ein String sein.")
    shelf = f.get("shelf_slope_db")
    if shelf is not None:
        # int oder float mit exaktem Wert 6 bzw. 12 (12.0 == 12); alles andere (bool, Str, NaN, 12.5) ist ungueltig
        if isinstance(shelf, bool) or not isinstance(shelf, (int, float)) or shelf not in (6, 12):
            raise FilterDigestError("shelf_slope_db muss None, 6 oder 12 sein.")
        shelf = int(shelf)  # kanonisch: int 6 bzw. 12
    return [ftype, canonical_float(f["fc_hz"]), canonical_float(f["gain_db"]), canonical_float(f["q"]), shelf]


def filters_digest(filters: Mapping) -> str:
    """SHA-256 (Hex) ueber die kanonische Form der drei Filterlisten."""
    if not isinstance(filters, dict):
        raise FilterDigestError("filters muss ein Objekt sein.")
    if set(filters) != set(FILTER_LISTS):
        raise FilterDigestError(f"filters muss genau die Listen {FILTER_LISTS} enthalten.")
    lists: dict[str, list[list[Any]]] = {}
    for name in FILTER_LISTS:
        lst = filters[name]
        if not isinstance(lst, list):
            raise FilterDigestError(f"filters.{name} muss eine Liste sein.")
        lists[name] = [_canonical_filter(f) for f in lst]
    payload = {"domain": DIGEST_DOMAIN, "lists": lists}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def peaking_filters_digest(filters_l: Any, filters_r: Any) -> str:
    """Digest fuer PeakingFilter-Listen (Attribute type, fc, gain_db, q) der KEF-Satelliten."""

    def _row(f: Any) -> dict[str, Any]:
        return {"type": f.type, "fc_hz": float(f.fc), "gain_db": float(f.gain_db), "q": float(f.q)}

    return filters_digest({
        "eq_proposals_l": [_row(f) for f in filters_l],
        "eq_proposals_r": [_row(f) for f in filters_r],
        "subwoofer": [],
    })
