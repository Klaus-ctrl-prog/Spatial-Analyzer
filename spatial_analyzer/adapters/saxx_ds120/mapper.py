"""Uebersetzt `filter_export.json` in ein Edit-JSON fuer den externen
`Candidate Builder 0.15.0-qa`.

Nur Uebersetzung/Quantisierung/Validierung - keine Akustik. Verwendet
ausschliesslich `filters.subwoofer` (nie eq_proposals_l/r), schreibt nur
Tabellenwerte ins Edit-JSON und dokumentiert jede Quantisierung.
Freigabe: nur `approval.status == RELEASED` (Do-No-Harm SAFE); der Mapper
erzeugt keine Datei und spricht keine Hardware an.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spatial_analyzer.adapters.saxx_ds120.quantization import (
    Quantized,
    SaxxAdapterError,
    load_lookup_tables,
    quantize,
)
from spatial_analyzer.adapters.saxx_ds120.validation import (
    BYPASS_ACTIVE,
    BYPASS_BYPASSED,
    EQ_BANDS,
    HARDWARE_VALIDATED,
    PHASE_STEPS_DEG,
    XOVER_TYPES,
    validate_edit_json,
)
from spatial_analyzer.filter_export import FilterExportError, validate_filter_export

TARGET_DEVICE = "saxx_ds120"
_SHELF_TYPE = {("lowshelf", 6): "LS6", ("lowshelf", 12): "LS12", ("highshelf", 6): "HS6", ("highshelf", 12): "HS12"}


class AdapterBlockedError(SaxxAdapterError):
    """Export ist nicht freigegeben (Do-No-Harm != SAFE)."""


class MappingError(SaxxAdapterError):
    pass


@dataclass
class MappingResult:
    edit: dict[str, Any]
    quantization: list[Quantized] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    released: bool = False
    hardware_validated: bool = HARDWARE_VALIDATED


def _assign_bands(filters: list[dict[str, Any]]) -> list[tuple[int, str, dict[str, Any]]]:
    """Shelves zuerst in EQ 1-9, danach PEQs fortlaufend (1-14)."""
    if len(filters) > EQ_BANDS:
        raise MappingError(f"{len(filters)} Subwoofer-Filter, aber nur {EQ_BANDS} EQ-Baender.")
    shelves, peqs = [], []
    for i, f in enumerate(filters):
        if f["type"] == "peq":
            peqs.append(("PEQ", f))
        else:
            slope = f.get("shelf_slope_db")
            if slope is None:
                raise MappingError(
                    f"subwoofer[{i}]: {f['type']} ohne shelf_slope_db (6/12) - Flanke wird nicht geraten."
                )
            shelves.append((_SHELF_TYPE[(f["type"], slope)], f))
    if len(shelves) > 9:
        raise MappingError(f"{len(shelves)} Shelf-Filter, aber Shelves sind nur in EQ 1-9 moeglich.")
    return [(n, t, f) for n, (t, f) in enumerate(shelves + peqs, start=1)]


def map_export_to_edit(
    doc: dict[str, Any],
    *,
    profile_name: str | None = None,
    require_released: bool = True,
    reset_unused_bands: bool = True,
    tables: dict[str, tuple[float, ...]] | None = None,
) -> MappingResult:
    """Erzeugt das Edit-JSON. `require_released=False` nur fuer die Vorschau
    (Ergebnis bleibt als nicht freigegeben markiert)."""
    try:
        validate_filter_export(doc)
    except FilterExportError as exc:
        raise MappingError(str(exc)) from exc
    if doc["target_device"] != TARGET_DEVICE:
        raise MappingError(f"target_device {doc['target_device']!r} ist nicht {TARGET_DEVICE!r}.")
    released = doc["approval"]["status"] == "RELEASED"
    why = f"Do-No-Harm: {doc['approval']['do_no_harm']}"
    if doc["approval"]["do_no_harm"] == "SAFE":
        why += "; Freigabe nicht an die exportierten Filter gebunden (Pruefung deckt sie nicht ab)"
    if require_released and not released:
        raise AdapterBlockedError(f"Export nicht freigegeben ({why}) - keine Abbildung.")
    tables = tables or load_lookup_tables()
    quant: list[Quantized] = []
    notes: list[str] = ["hardware_validated=false: Offline-Abbildung ist keine Hardwarevalidierung."]
    edit: dict[str, Any] = {}
    if not released:
        notes.append(f"NICHT FREIGEGEBEN ({why}) - nur Vorschau.")

    if profile_name is not None:
        edit["name"] = profile_name

    xo = doc["crossovers"]
    if "phase_deg" in xo:
        if xo["phase_deg"] not in PHASE_STEPS_DEG:
            raise MappingError(f"phase_deg {xo['phase_deg']} nicht in 45-Grad-Stufen {PHASE_STEPS_DEG}.")
        edit["phase_deg"] = xo["phase_deg"]
    for key in ("hpf", "lpf"):
        x = xo[key]
        if x is None:
            continue
        if x["type"] not in XOVER_TYPES:
            raise MappingError(f"{key}: Weichentyp {x['type']!r} nicht unterstuetzt ({XOVER_TYPES}).")
        q = quantize(tables["frequency_hz"], x["frequency_hz"], f"{key.upper()} Frequenz [Hz]")
        quant.append(q)
        edit[key] = {"frequency_hz": q.table_value, "type": x["type"], "enabled": x["enabled"]}
        notes.append(
            f"{key.upper()}: Bypass-Flag {BYPASS_ACTIVE if x['enabled'] else BYPASS_BYPASSED} "
            f"({'aktiv' if x['enabled'] else 'Bypass'})."
        )

    eq: list[dict[str, Any]] = []
    used = set()
    for band, dev_type, f in _assign_bands(doc["filters"]["subwoofer"]):
        qf = quantize(tables["frequency_hz"], f["fc_hz"], f"EQ{band} Frequenz [Hz]")
        qg = quantize(tables["gain_db"], f["gain_db"], f"EQ{band} Gain [dB]")
        qq = quantize(tables["q"], f["q"], f"EQ{band} Q")
        quant += [qf, qg, qq]
        item: dict[str, Any] = {"band": band, "frequency_hz": qf.table_value, "gain_db": qg.table_value, "q": qq.table_value}
        if band <= 9:
            item["type"] = dev_type
            item["bypass"] = False
        eq.append(item)
        used.add(band)
    if reset_unused_bands:
        zero = quantize(tables["gain_db"], 0.0, "EQ ungenutzt Gain [dB]")
        if zero.table_value != 0.0:
            raise MappingError("Gain-Tabelle enthaelt keinen Wert 0 dB.")
        for band in range(1, EQ_BANDS + 1):
            if band in used:
                continue
            # 1-9 haben ein Bypass-Feld; 10-14 nicht -> dort Gain 0 dB.
            eq.append({"band": band, "bypass": True} if band <= 9 else {"band": band, "gain_db": 0.0})
        notes.append("Ungenutzte EQ-Baender: 1-9 bypass, 10-14 Gain 0 dB (kein Bypass-Feld).")
    else:
        notes.append("Ungenutzte EQ-Baender bleiben wie im Template (kein Reset).")
    eq.sort(key=lambda e: e["band"])
    if eq:
        edit["eq"] = eq

    validate_edit_json(edit, tables)
    return MappingResult(edit=edit, quantization=quant, notes=notes, released=released)
