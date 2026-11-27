"""Uebersetzt `filter_export.json` in ein Edit-JSON fuer den externen
`Candidate Builder 0.15.0-qa`.

Nur Uebersetzung/Quantisierung/Validierung - keine Akustik. Verwendet
ausschliesslich `filters.subwoofer` (nie eq_proposals_l/r), schreibt nur
Tabellenwerte ins Edit-JSON und dokumentiert jede Quantisierung.
Freigabe: Der Saxx DS120 ist IMMER nicht freigegeben (kein Subwoofer-Do-No-Harm-
Modell, siehe filter_export.is_device_released); die Abbildung ist nur als
Vorschau moeglich. Der Mapper erzeugt keine Datei und spricht keine Hardware an.

Reproduzierbarkeit: Eingabe- (mapping_input_sha256), Ausgabe- (mapping_output_sha256)
und Tabellen-Digest (tables_sha256) werden kanonisch berechnet. verify_mapping()
rechnet neu und vergleicht; ein separat gelieferter Hash wird nie vertraut.
Integritaet, keine Authentizitaet (kein Geheimnis/keine Signatur).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from spatial_analyzer.adapters.saxx_ds120.quantization import (
    Quantized,
    SaxxAdapterError,
    load_lookup_tables,
    quantize,
)
from spatial_analyzer.adapters.saxx_ds120.validation import (
    BUILDER_VERSION,
    BYPASS_ACTIVE,
    BYPASS_BYPASSED,
    EQ_BANDS,
    HARDWARE_VALIDATED,
    PHASE_STEPS_DEG,
    XOVER_TYPES,
    validate_edit_json,
)
from spatial_analyzer.filter_digest import FilterDigestError, canonical_float
from spatial_analyzer.filter_export import (
    FilterExportError,
    is_device_released,
    validate_filter_export,
)

TARGET_DEVICE = "saxx_ds120"
MAPPER_VERSION = "1"
_SHELF_TYPE = {("lowshelf", 6): "LS6", ("lowshelf", 12): "LS12", ("highshelf", 6): "HS6", ("highshelf", 12): "HS12"}
_TABLE_NAMES = ("frequency_hz", "gain_db", "q")


class AdapterBlockedError(SaxxAdapterError):
    """Export ist nicht freigegeben (Saxx: keine Geraetefreigabe ohne Subwoofer-Do-No-Harm-Modell)."""


class MappingError(SaxxAdapterError):
    pass


@dataclass
class MappingResult:
    """Ergebnis der Abbildung. `notes` sind rein informativ und NICHT vom mapping_output_sha256 erfasst."""

    edit: dict[str, Any]
    quantization: list[Quantized] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    released: bool = False
    hardware_validated: bool = HARDWARE_VALIDATED
    mapping_input_sha256: str = ""
    mapping_output_sha256: str = ""
    tables_sha256: str = ""


def _canon_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _canon_tree(v: Any) -> Any:
    """Rekursiv: Floats normiert (-0.0 -> 0.0), ganze Zahlen/bool/str/None unveraendert."""
    if isinstance(v, dict):
        return {k: _canon_tree(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon_tree(x) for x in v]
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return canonical_float(v)
    raise MappingError(f"Nicht kanonisierbarer Wert vom Typ {type(v).__name__}.")


def tables_sha256(tables: dict[str, Any]) -> str:
    """SHA-256 der tatsaechlich verwendeten Tabellen (frequency_hz, gain_db, q)."""
    try:
        payload = {name: [canonical_float(v) for v in tables[name]] for name in _TABLE_NAMES}
        text = _canon_json(payload)
    except (FilterDigestError, KeyError, TypeError) as exc:
        raise MappingError(f"Tabellen nicht digestierbar ({exc!r}).") from exc
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canon_shelf(slope: Any) -> int | None:
    """Shelf-Flanke kanonisch: None oder exakt 6/12 (int oder float, 12.0 == 12); alles andere ist ein Fehler."""
    if slope is None:
        return None
    if isinstance(slope, bool) or not isinstance(slope, (int, float)) or slope not in (6, 12):
        raise MappingError(f"shelf_slope_db {slope!r} ist keine gueltige Flanke (6 oder 12).")
    return int(slope)


def compute_mapping_input_sha256(
    doc: dict[str, Any],
    *,
    profile_name: str | None,
    reset_unused_bands: bool,
    tables: dict[str, tuple[float, ...]] | None,
) -> str:
    """SHA-256 aller Eingaben, die das Edit-JSON bestimmen (Export, Optionen, Tabellen)."""
    tbl = tables if tables is not None else load_lookup_tables()
    cf = canonical_float
    try:
        xo = doc["crossovers"]

        def _xo(x: Any) -> dict[str, Any] | None:
            if x is None:
                return None
            return {"frequency_hz": cf(x["frequency_hz"]), "type": x["type"], "enabled": x["enabled"]}

        payload = {
            "domain": "spatial_analyzer.saxx_ds120.mapping_input.v1",
            "mapper_version": MAPPER_VERSION,
            "builder_version": BUILDER_VERSION,
            "contract_version": doc["contract_version"],
            "target_device": doc["target_device"],
            "subwoofer": [
                [f["type"], cf(f["fc_hz"]), cf(f["gain_db"]), cf(f["q"]), _canon_shelf(f.get("shelf_slope_db"))]
                for f in doc["filters"]["subwoofer"]
            ],
            "crossovers": {"hpf": _xo(xo["hpf"]), "lpf": _xo(xo["lpf"]), "phase_deg": xo.get("phase_deg")},
            "profile_name": profile_name,
            "reset_unused_bands": bool(reset_unused_bands),
            "tables_sha256": tables_sha256(tbl),
        }
        text = _canon_json(payload)
    except (FilterDigestError, KeyError, TypeError) as exc:
        raise MappingError(f"Mapping-Eingabe nicht digestierbar ({exc!r}).") from exc
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_mapping_output_sha256(result_like: Any) -> str:
    """SHA-256 des Ergebnisses (edit, Quantisierung, Status, Bezug zu Eingabe und Tabellen)."""
    cf = canonical_float
    try:
        payload = {
            "domain": "spatial_analyzer.saxx_ds120.mapping_output.v1",
            "mapping_input_sha256": result_like.mapping_input_sha256,
            "edit": _canon_tree(result_like.edit),
            "quantization": [
                [q.label, cf(q.target), cf(q.table_value), q.index, cf(q.deviation)]
                for q in result_like.quantization
            ],
            "released": bool(result_like.released),
            "hardware_validated": bool(result_like.hardware_validated),
            "tables_sha256": result_like.tables_sha256,
        }
        text = _canon_json(payload)
    except (FilterDigestError, AttributeError, TypeError) as exc:
        raise MappingError(f"Mapping-Ergebnis nicht digestierbar ({exc!r}).") from exc
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    """Erzeugt das Edit-JSON. Saxx ist nie geraetefreigegeben: mit
    `require_released=True` (Standard) wird immer AdapterBlockedError geworfen;
    `require_released=False` liefert die als nicht freigegeben markierte Vorschau."""
    try:
        validate_filter_export(doc)
    except FilterExportError as exc:
        raise MappingError(str(exc)) from exc
    if doc["target_device"] != TARGET_DEVICE:
        raise MappingError(f"target_device {doc['target_device']!r} ist nicht {TARGET_DEVICE!r}.")
    released = is_device_released(doc)  # fuer Saxx immer False
    why = f"Do-No-Harm: {doc['approval']['do_no_harm']}"
    if doc["approval"]["do_no_harm"] == "SAFE":
        why += "; Freigabe nicht an die exportierten Filter gebunden (Pruefung deckt sie nicht ab)"
    if require_released and not released:
        raise AdapterBlockedError(
            f"Export nicht freigegeben ({why}) - keine Abbildung. "
            "Saxx DS120: keine Geraetefreigabe ohne Subwoofer-Do-No-Harm-Modell."
        )
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
    result = MappingResult(edit=edit, quantization=quant, notes=notes, released=released)
    result.mapping_input_sha256 = compute_mapping_input_sha256(
        doc, profile_name=profile_name, reset_unused_bands=reset_unused_bands, tables=tables
    )
    result.tables_sha256 = tables_sha256(tables)
    result.mapping_output_sha256 = compute_mapping_output_sha256(result)
    return result


def verify_mapping(
    doc: dict[str, Any],
    result: MappingResult,
    *,
    profile_name: str | None = None,
    reset_unused_bands: bool = True,
    tables: dict[str, tuple[float, ...]] | None = None,
) -> None:
    """Rechnet die Abbildung neu und prueft das Ergebnis; wirft MappingError bei Abweichung.

    Hinweis: `notes` sind informativ und werden nicht verifiziert (nicht vom Ausgabe-Hash erfasst).

    Vertraut keinem mitgelieferten Hash: verglichen werden die Neuberechnung
    (Eingabe-Hash, Edit, Quantisierung, Ausgabe-Hash) und der Ausgabe-Hash aus
    dem aktuellen Inhalt von `result` (erkennt nachtraegliche Aenderungen).
    """
    try:
        fresh = map_export_to_edit(
            doc, profile_name=profile_name, require_released=False,
            reset_unused_bands=reset_unused_bands, tables=tables,
        )
    except MappingError:
        raise
    except SaxxAdapterError as exc:
        raise MappingError(f"Neuberechnung fehlgeschlagen: {exc}") from exc
    problems: list[str] = []
    if fresh.mapping_input_sha256 != result.mapping_input_sha256:
        problems.append("mapping_input_sha256 weicht ab (Export, Profilname, Optionen oder Tabellen geaendert)")
    if fresh.edit != result.edit:
        problems.append("edit weicht von der Neuberechnung ab")
    if fresh.quantization != result.quantization:
        problems.append("quantization weicht von der Neuberechnung ab")
    if fresh.released != result.released:
        problems.append("released weicht von der Neuberechnung ab")
    if fresh.mapping_output_sha256 != result.mapping_output_sha256:
        problems.append("mapping_output_sha256 weicht von der Neuberechnung ab")
    if compute_mapping_output_sha256(result) != result.mapping_output_sha256:
        problems.append("mapping_output_sha256 passt nicht zu edit/quantization des Ergebnisses (manipuliert)")
    if result.hardware_validated is not False:
        problems.append("hardware_validated muss false sein")
    if problems:
        raise MappingError("Mapping-Verifikation fehlgeschlagen:\n- " + "\n- ".join(problems))
