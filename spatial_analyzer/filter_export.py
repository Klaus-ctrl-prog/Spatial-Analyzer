"""Neutraler Austauschvertrag `filter_export.json` (contract_version 1.0).

Vertragsgrundlage: AGENTS.md 4.1 (Projekt-Verfassung); allgemeine
Projektspezifikation: docs/SPECIFICATION.md (definiert diesen Vertrag nicht
im Detail - massgeblich ist das Schema unter adapters/schemas/).

Einmal messen, getrennt adaptieren: Der Kern (eq_solver.py/pipeline.py)
berechnet die Filter; dieses Modul serialisiert sie geraeteneutral und
prueft den Vertrag. Geraete-Adapter (spatial_analyzer/adapters/...) lesen
nur diese Datei.

Harte Regeln:
- KEF-L/R-Satellitenfilter (`filters.eq_proposals_l/r`) und Subwoofer-Filter
  (`filters.subwoofer`) sind getrennte Listen. Satelliten-PEQs werden NIE
  fuer ein Subwoofer-Geraet exportiert; Subwoofer-Filter muessen explizit
  uebergeben werden und werden nie aus eq_proposals_* abgeleitet.
- Do-No-Harm-Status (SAFE/UNSAFE/INCONCLUSIVE) wird unveraendert uebernommen.
  `approval.status` ist nur `RELEASED`, wenn (a) der Status SAFE ist UND (b) die
  exportierten Filter genau die Filter sind, auf die sich die Pruefung bezog:
  `approval.checked_filters_sha256` (Digest der geprueften Filter) muss dem
  Digest der exportierten `filters` entsprechen (compute_filters_digest).
  Die Pipeline prueft nur die KEF-Satellitenfilter; Subwoofer-Filter sind
  daher nur freigegeben, wenn der Aufrufer ausdruecklich die von der Pruefung
  abgedeckten Subwoofer-Filter uebergibt (`checked_subwoofer_filters`).
  Der Digest ist ein Integritaets-, kein Authentizitaetsnachweis: er erkennt
  nachtraegliche Aenderungen der Filter, schuetzt aber nicht vor gezielter
  Faelschung (kein Geheimnis/keine Signatur).
- `approval.hardware_validated` ist immer `false` (Hardware-Gate).
- Quelldaten werden nicht veraendert; Ausgabe ist deterministisch.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from spatial_analyzer.biquad_preview import PeakingFilter
from spatial_analyzer.eq_solver import EqProposal, NoHarmResult, NoHarmStatus

CONTRACT_VERSION = "1.0"
TARGET_DEVICES = ("saxx_ds120", "mosconi_pico_8_10", "kef_ls50_wireless")
SATELLITE_TARGETS = ("kef_ls50_wireless",)  # nur hier sind eq_proposals_l/r zulaessig
EXPORTABLE_FILTER_TYPES = ("peq", "lowshelf", "highshelf")
_TOP_KEYS = {
    "contract_version", "run_id", "target_device", "approval", "intentional_flat",
    "filters", "crossovers", "provenance",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")  # nur mit fullmatch verwenden ('$' wuerde "\n" am Ende zulassen)
_FILTER_LISTS = ("eq_proposals_l", "eq_proposals_r", "subwoofer")
_APPROVAL_KEYS = {"status", "do_no_harm", "hardware_validated", "checked_filters_sha256"}


def _is_sha256(v: Any) -> bool:
    return isinstance(v, str) and _SHA256_RE.fullmatch(v) is not None


def compute_filters_digest(filters: dict[str, Any]) -> str:
    """SHA-256 ueber die kanonische Form der Filterlisten.

    Erfasst je Filter nur, was die Do-No-Harm-Vorschau auswertet: type, fc_hz,
    gain_db, q (Zahlen normiert auf float; `reason` und `shelf_slope_db`
    zaehlen nicht) - in Listenreihenfolge, getrennt nach den drei Listen.
    """
    if not isinstance(filters, dict):
        raise FilterExportError("filters muss ein Objekt sein.")
    canon: dict[str, list[list[Any]]] = {}
    for name in _FILTER_LISTS:
        lst = filters.get(name)
        if not isinstance(lst, list):
            raise FilterExportError(f"filters.{name} muss eine Liste sein.")
        rows = []
        for f in lst:
            try:
                rows.append([f["type"], float(f["fc_hz"]), float(f["gain_db"]), float(f["q"])])
            except (KeyError, TypeError, ValueError) as exc:
                raise FilterExportError(f"filters.{name}: Filter nicht digestierbar ({exc!r}).") from exc
        canon[name] = rows
    text = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FilterExportError(ValueError):
    """Ungueltiger Export bzw. ungueltige Eingabe (alle Befunde in der Meldung)."""


def _filter_to_dict(f: PeakingFilter, reason: str | None = None, shelf_slope_db: int | None = None) -> dict[str, Any]:
    if f.type not in EXPORTABLE_FILTER_TYPES:
        raise FilterExportError(f"Filtertyp {f.type!r} ist nicht exportierbar (erlaubt: {EXPORTABLE_FILTER_TYPES}).")
    d: dict[str, Any] = {"type": f.type, "fc_hz": float(f.fc), "gain_db": float(f.gain_db), "q": float(f.q)}
    if shelf_slope_db is not None:
        d["shelf_slope_db"] = shelf_slope_db
    if reason:
        d["reason"] = reason
    return d


def build_filter_export(
    *,
    run_id: str,
    target_device: str,
    no_harm: NoHarmResult,
    measurement_set_id: str,
    source_profile_sha256: str | None = None,
    eq_proposals_l: Iterable[EqProposal] | None = None,
    eq_proposals_r: Iterable[EqProposal] | None = None,
    subwoofer_filters: Iterable[PeakingFilter] | None = None,
    subwoofer_shelf_slopes_db: dict[int, int] | None = None,
    checked_subwoofer_filters: Iterable[PeakingFilter] | None = None,
    crossovers: dict[str, Any] | None = None,
    intentional_flat: bool = False,
) -> dict[str, Any]:
    """Baut und validiert ein Export-Dokument.

    `subwoofer_shelf_slopes_db` ordnet Shelf-Filtern (Index in
    `subwoofer_filters`) die Flanke 6 oder 12 dB zu; ohne Angabe bleibt sie
    offen und der Saxx-Adapter verweigert Shelves (keine geratene Flanke).

    Freigabebindung: `no_harm` gilt fuer die Filter, die die Pruefung
    ausgewertet hat - bei der Pipeline die Satellitenfilter. Subwoofer-Filter
    sind nur `RELEASED`, wenn `checked_subwoofer_filters` (die von `no_harm`
    abgedeckten Subwoofer-Filter) denselben Digest ergibt wie
    `subwoofer_filters`; sonst bleibt der Export `BLOCKED`.
    """
    sat_l = list(eq_proposals_l or [])
    sat_r = list(eq_proposals_r or [])
    sub = list(subwoofer_filters or [])
    if target_device not in TARGET_DEVICES:
        raise FilterExportError(f"Unbekanntes target_device {target_device!r}; erlaubt: {TARGET_DEVICES}.")
    if target_device not in SATELLITE_TARGETS and (sat_l or sat_r):
        raise FilterExportError(
            "KEF-Satellitenfilter (eq_proposals_l/r) duerfen nicht fuer "
            f"{target_device} exportiert werden; Subwoofer-Filter muessen explizit uebergeben werden."
        )
    if target_device in SATELLITE_TARGETS and sub:
        raise FilterExportError("Subwoofer-Filter duerfen nicht in einen KEF-Satelliten-Export.")
    slopes = subwoofer_shelf_slopes_db or {}
    bad = [i for i in slopes if not 0 <= i < len(sub)]
    if bad:
        raise FilterExportError(f"subwoofer_shelf_slopes_db verweist auf nicht vorhandene Filter-Indizes {bad}.")
    xo = copy.deepcopy(crossovers) if crossovers else {}  # keine Referenzen auf Aufrufer-Daten
    checked_digest = compute_filters_digest({
        "eq_proposals_l": [_filter_to_dict(p.filter) for p in sat_l],
        "eq_proposals_r": [_filter_to_dict(p.filter) for p in sat_r],
        "subwoofer": [_filter_to_dict(f) for f in list(checked_subwoofer_filters or [])],
    })
    doc: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "target_device": target_device,
        "approval": {
            "status": "BLOCKED",  # unten gesetzt, sobald der Digest der exportierten Filter feststeht
            "do_no_harm": no_harm.status.value,
            "hardware_validated": False,
            "checked_filters_sha256": checked_digest,
        },
        "intentional_flat": bool(intentional_flat),
        "filters": {
            "eq_proposals_l": [_filter_to_dict(p.filter, p.reason) for p in sat_l],
            "eq_proposals_r": [_filter_to_dict(p.filter, p.reason) for p in sat_r],
            "subwoofer": [_filter_to_dict(f, shelf_slope_db=slopes.get(i)) for i, f in enumerate(sub)],
        },
        "crossovers": {
            "hpf": xo.get("hpf"),
            "lpf": xo.get("lpf"),
            **({"phase_deg": xo["phase_deg"]} if "phase_deg" in xo else {}),
        },
        "provenance": {"measurement_set_id": measurement_set_id, "source_profile_sha256": source_profile_sha256},
    }
    if no_harm.status is NoHarmStatus.SAFE and compute_filters_digest(doc["filters"]) == checked_digest:
        doc["approval"]["status"] = "RELEASED"
    return validate_filter_export(doc)


def _finite(v: Any) -> bool:
    """math.isfinite ohne OverflowError bei riesigen ganzen Zahlen."""
    try:
        return math.isfinite(v)
    except (OverflowError, TypeError):
        return False


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and _finite(v)


def _check_filter(f: Any, where: str, problems: list[str]) -> None:
    if not isinstance(f, dict):
        problems.append(f"{where}: muss ein Objekt sein.")
        return
    allowed = {"type", "fc_hz", "gain_db", "q", "shelf_slope_db", "reason"}
    for k in sorted(set(f) - allowed):
        problems.append(f"{where}: unbekanntes Feld {k!r}.")
    for k in ("type", "fc_hz", "gain_db", "q"):
        if k not in f:
            problems.append(f"{where}: Pflichtfeld {k!r} fehlt.")
    if "type" in f and f["type"] not in EXPORTABLE_FILTER_TYPES:
        problems.append(f"{where}: type {f['type']!r} nicht erlaubt {EXPORTABLE_FILTER_TYPES}.")
    for k in ("fc_hz", "q"):
        if k in f and not (_num(f[k]) and f[k] > 0):
            problems.append(f"{where}: {k} muss eine endliche Zahl > 0 sein.")
    if "gain_db" in f and not _num(f["gain_db"]):
        problems.append(f"{where}: gain_db muss eine endliche Zahl sein.")
    if "shelf_slope_db" in f:
        s = f["shelf_slope_db"]
        if isinstance(s, bool) or s not in (6, 12):
            problems.append(f"{where}: shelf_slope_db muss 6 oder 12 sein.")
        if f.get("type") == "peq":
            problems.append(f"{where}: shelf_slope_db ist fuer peq unzulaessig.")
    if "reason" in f and not isinstance(f["reason"], str):
        problems.append(f"{where}: reason muss ein String sein.")


def _check_crossover(x: Any, where: str, problems: list[str]) -> None:
    if x is None:
        return
    if not isinstance(x, dict):
        problems.append(f"{where}: muss null oder ein Objekt sein.")
        return
    for k in sorted(set(x) - {"frequency_hz", "type", "enabled"}):
        problems.append(f"{where}: unbekanntes Feld {k!r}.")
    for k in ("frequency_hz", "type", "enabled"):
        if k not in x:
            problems.append(f"{where}: Pflichtfeld {k!r} fehlt.")
    if "frequency_hz" in x and not (_num(x["frequency_hz"]) and x["frequency_hz"] > 0):
        problems.append(f"{where}: frequency_hz muss eine endliche Zahl > 0 sein.")
    if "type" in x and not isinstance(x["type"], str):
        problems.append(f"{where}: type muss ein String sein.")
    if "enabled" in x and not isinstance(x["enabled"], bool):
        problems.append(f"{where}: enabled muss true/false sein.")


def validate_filter_export(doc: Any) -> dict[str, Any]:
    """Prueft den Vertrag vollstaendig; gibt `doc` unveraendert zurueck oder
    wirft FilterExportError mit allen Befunden."""
    problems: list[str] = []
    if not isinstance(doc, dict):
        raise FilterExportError("filter_export muss ein JSON-Objekt sein.")
    for k in sorted(_TOP_KEYS - set(doc)):
        problems.append(f"Pflichtfeld {k!r} fehlt.")
    for k in sorted(set(doc) - _TOP_KEYS):
        problems.append(f"Unbekanntes Feld {k!r}.")
    if "contract_version" in doc and doc["contract_version"] != CONTRACT_VERSION:
        problems.append(f"contract_version muss {CONTRACT_VERSION!r} sein (ist {doc['contract_version']!r}).")
    if "run_id" in doc and not (isinstance(doc["run_id"], str) and doc["run_id"].strip()):
        problems.append("run_id muss ein nicht leerer String sein.")
    device = doc.get("target_device")
    if "target_device" in doc and device not in TARGET_DEVICES:
        problems.append(f"target_device {device!r} unbekannt; erlaubt: {TARGET_DEVICES}.")

    ap = doc.get("approval")
    if "approval" in doc:
        if not isinstance(ap, dict):
            problems.append("approval muss ein Objekt sein.")
        else:
            for k in sorted(set(ap) ^ _APPROVAL_KEYS):
                problems.append(f"approval: Feld {k!r} fehlt oder ist unbekannt.")
            dnh, st = ap.get("do_no_harm"), ap.get("status")
            if not (isinstance(dnh, str) and dnh in {s.value for s in NoHarmStatus}):
                problems.append("approval.do_no_harm muss SAFE, UNSAFE oder INCONCLUSIVE sein.")
            if not (isinstance(st, str) and st in ("RELEASED", "BLOCKED")):
                problems.append("approval.status muss RELEASED oder BLOCKED sein.")
            elif st == "RELEASED" and dnh != "SAFE":
                problems.append("approval.status RELEASED ist nur bei do_no_harm SAFE zulaessig.")
            if "checked_filters_sha256" in ap and not _is_sha256(ap["checked_filters_sha256"]):
                problems.append("approval.checked_filters_sha256 muss 64 Zeichen Kleinbuchstaben-Hex sein.")
            if ap.get("hardware_validated") is not False:
                problems.append("approval.hardware_validated muss false sein (Hardware-Gate bis zum physischen Rundtest).")

    if "intentional_flat" in doc and not isinstance(doc["intentional_flat"], bool):
        problems.append("intentional_flat muss true/false sein.")

    fl = doc.get("filters")
    n_filters = 0
    if "filters" in doc:
        if not isinstance(fl, dict):
            problems.append("filters muss ein Objekt sein.")
        else:
            for k in sorted(set(fl) ^ {"eq_proposals_l", "eq_proposals_r", "subwoofer"}):
                problems.append(f"filters: Feld {k!r} fehlt oder ist unbekannt.")
            for k in ("eq_proposals_l", "eq_proposals_r", "subwoofer"):
                if k not in fl:
                    continue
                lst = fl[k]
                if not isinstance(lst, list):
                    problems.append(f"filters.{k} muss eine Liste sein.")
                    continue
                n_filters += len(lst)
                for i, f in enumerate(lst):
                    _check_filter(f, f"filters.{k}[{i}]", problems)
            if device in TARGET_DEVICES:
                if device not in SATELLITE_TARGETS and (fl.get("eq_proposals_l") or fl.get("eq_proposals_r")):
                    problems.append(f"KEF-Satellitenfilter duerfen nicht fuer {device} exportiert werden.")
                if device in SATELLITE_TARGETS and fl.get("subwoofer"):
                    problems.append("Subwoofer-Filter duerfen nicht in einen KEF-Satelliten-Export.")

    xo = doc.get("crossovers")
    has_xo = False
    if "crossovers" in doc:
        if not isinstance(xo, dict):
            problems.append("crossovers muss ein Objekt sein.")
        else:
            for k in sorted(set(xo) - {"hpf", "lpf", "phase_deg"}):
                problems.append(f"crossovers: unbekanntes Feld {k!r}.")
            for k in ("hpf", "lpf"):
                if k not in xo:
                    problems.append(f"crossovers: Pflichtfeld {k!r} fehlt (null = nicht gesetzt).")
                _check_crossover(xo.get(k), f"crossovers.{k}", problems)
            if "phase_deg" in xo and (isinstance(xo["phase_deg"], bool) or not isinstance(xo["phase_deg"], int)):
                problems.append("crossovers.phase_deg muss eine ganze Zahl (Grad) sein.")
            has_xo = bool(xo.get("hpf") or xo.get("lpf") or "phase_deg" in xo)

    if isinstance(doc.get("intentional_flat"), bool) and isinstance(fl, dict):
        if doc["intentional_flat"] and n_filters:
            problems.append("intentional_flat=true widerspricht vorhandenen Filtern.")
        if not doc["intentional_flat"] and n_filters == 0 and not has_xo:
            problems.append("Leerer Export (keine Filter/Weichen) muss mit intentional_flat=true gekennzeichnet sein.")

    pv = doc.get("provenance")
    if "provenance" in doc:
        if not isinstance(pv, dict):
            problems.append("provenance muss ein Objekt sein.")
        else:
            for k in sorted(set(pv) ^ {"measurement_set_id", "source_profile_sha256"}):
                problems.append(f"provenance: Feld {k!r} fehlt oder ist unbekannt.")
            m = pv.get("measurement_set_id")
            if "measurement_set_id" in pv and not (isinstance(m, str) and m.strip()):
                problems.append("provenance.measurement_set_id muss ein nicht leerer String sein.")
            h = pv.get("source_profile_sha256")
            if "source_profile_sha256" in pv and h is not None and not _is_sha256(h):
                problems.append("provenance.source_profile_sha256 muss null oder 64 Zeichen Kleinbuchstaben-Hex sein.")

    # Freigabebindung: RELEASED nur, wenn die geprueften Filter den exportierten entsprechen.
    if (
        not problems
        and doc["approval"]["status"] == "RELEASED"
        and doc["approval"]["checked_filters_sha256"] != compute_filters_digest(doc["filters"])
    ):
        problems.append(
            "approval.status RELEASED, aber checked_filters_sha256 entspricht nicht den exportierten Filtern "
            "(Filter nach der Pruefung geaendert oder nicht von der Pruefung abgedeckt)."
        )

    if problems:
        raise FilterExportError("Ungueltiger filter_export:\n- " + "\n- ".join(problems))
    return doc


def dumps_filter_export(doc: dict[str, Any]) -> str:
    """Deterministische Serialisierung (sortierte Schluessel, 2 Leerzeichen, LF)."""
    validate_filter_export(doc)
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_filter_export(path: Path | str, doc: dict[str, Any], *, overwrite: bool = False) -> Path:
    p = Path(path)
    text = dumps_filter_export(doc)  # validiert vor dem Schreiben
    if p.exists() and not overwrite:
        raise FilterExportError(f"Verweigere Ueberschreiben von {p} (overwrite=True setzen).")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


def load_filter_export(path: Path | str) -> dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FilterExportError(f"filter_export nicht lesbar: {exc}") from exc
    return validate_filter_export(doc)
