"""Nachvollziehbare Quantisierung auf die originalen Saxx-DS120-Lookup-Tabellen.

Die Tabellen (244 Frequenz-, 181 Gain-, 79 Q-Stufen) stammen unveraendert aus
dem Saxx-Validierungsprojekt; ihr SHA-256 wird bei jedem Laden geprueft.
Algorithmus identisch zum externen `Candidate Builder 0.15.0-qa`: naechster
Tabellenwert, bei Gleichstand der erste Index, KEIN Randklemmen (ausserhalb
des Tabellenbereichs -> Fehler).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

LOOKUP_FILE = Path(__file__).with_name("saxx_lookup_tables.json")
LOOKUP_SHA256 = "7a9b03ffe62d42068d612e1cc3ff078338fad79317490b8ebf599f6b188d85e6"
EXPECTED_COUNTS = {"frequency_hz": 244, "gain_db": 181, "q": 79}


def _finite(v: object) -> bool:
    """math.isfinite ohne OverflowError bei riesigen ganzen Zahlen."""
    try:
        return math.isfinite(v)
    except (OverflowError, TypeError):
        return False


class SaxxAdapterError(ValueError):
    """Basisklasse aller Fehler des Saxx-DS120-Adapters."""


class QuantizationError(SaxxAdapterError):
    pass


@dataclass(frozen=True)
class Quantized:
    label: str
    target: float  # Sollwert
    table_value: float  # Tabellenwert
    index: int
    deviation: float  # table_value - target

    def describe(self) -> str:
        return f"{self.label}: {self.target:g} -> {self.table_value:g} (Index {self.index}, Abweichung {self.deviation:+.4g})"


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, tuple[float, ...]]:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != LOOKUP_SHA256:
        raise QuantizationError("Lookup-Datei hat nicht den erwarteten SHA-256 - Verweigerung.")
    doc = json.loads(raw.decode("utf-8"))
    tables: dict[str, tuple[float, ...]] = {}
    for name, count in EXPECTED_COUNTS.items():
        vals = doc["tables"][name]["values"]
        if len(vals) != count:
            raise QuantizationError(f"Tabelle {name}: {len(vals)} statt {count} Stufen.")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vals):
            raise QuantizationError(f"Tabelle {name} enthaelt nicht-endliche/nicht-numerische Werte.")
        if any(a >= b for a, b in zip(vals, vals[1:])):
            raise QuantizationError(f"Tabelle {name} ist nicht streng monoton steigend.")
        tables[name] = tuple(float(v) for v in vals)
    return tables


def load_lookup_tables(path: Path | str | None = None) -> dict[str, tuple[float, ...]]:
    """Laedt die Tabellen (Standard: gebuendelte Datei) mit Hash- und Strukturpruefung."""
    return _load(str(Path(path) if path is not None else LOOKUP_FILE))


def quantize(table: tuple[float, ...] | list[float], target: float, label: str) -> Quantized:
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not _finite(target):
        raise QuantizationError(f"{label}: Sollwert muss eine endliche Zahl sein.")
    t = float(target)
    if t < table[0] or t > table[-1]:
        raise QuantizationError(
            f"{label}={t:g} liegt ausserhalb des Saxx-Tabellenbereichs [{table[0]:g}, {table[-1]:g}] (kein Randklemmen)."
        )
    idx = min(range(len(table)), key=lambda i: abs(table[i] - t))  # erster Index bei Gleichstand
    return Quantized(label, t, float(table[idx]), idx, float(table[idx]) - t)
