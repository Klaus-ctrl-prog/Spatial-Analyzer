"""Tests fuer filter_export.py (Vertrag 1.1, Legacy 1.0 nur lesend; synthetische Daten)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from spatial_analyzer.adapters.saxx_ds120 import AdapterBlockedError, build_preview, map_export_to_edit
from spatial_analyzer.biquad_preview import PeakingFilter
from spatial_analyzer.eq_solver import (
    CHECKED_SCOPE_LEFT_RIGHT,
    CHECKED_SCOPE_NONE,
    EqProposal,
    NoHarmResult,
    NoHarmStatus,
    validate_no_harm,
)
from spatial_analyzer.filter_digest import filters_digest, peaking_filters_digest
from spatial_analyzer.filter_export import (
    FilterExportError,
    build_filter_export,
    compute_filters_digest,
    dumps_filter_export,
    is_device_released,
    load_filter_export,
    validate_filter_export,
    write_filter_export,
)

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "spatial_analyzer" / "adapters" / "schemas"
SCHEMA = SCHEMA_DIR / "filter_export.schema.json"
SCHEMA_10 = SCHEMA_DIR / "filter_export_1_0.schema.json"
SAFE = NoHarmResult(NoHarmStatus.SAFE, "ok", 5)  # von Hand gebaut: ohne Bindung an Filter


def _real_no_harm(filters_l, filters_r):
    """Echte Do-No-Harm-Pruefung auf synthetischen Daten (3 Positionen, leichte Teilausloeschung)."""
    freq = np.linspace(90.0, 110.0, 21)
    phase = np.deg2rad(180.0 + np.array([0.0, -5.0, 5.0]))[:, None]
    l_complex = np.ones((3, freq.size), dtype=np.complex128)
    r_complex = 0.5 * np.exp(1j * phase) * np.ones((3, freq.size))
    return validate_no_harm(freq, ["p1", "p2", "p3"], l_complex, r_complex, filters_l, filters_r)


def _sub_doc(**kw):
    args = dict(
        run_id="r1", target_device="saxx_ds120", no_harm=SAFE, measurement_set_id="m1",
        subwoofer_filters=[PeakingFilter("peq", 50.0, -3.0, 4.0)],
    )
    args.update(kw)
    return build_filter_export(**args)


def _kef_doc(filters=None, no_harm=None, **kw):
    """KEF-Export; ohne no_harm wird die echte Pruefung ueber identische L/R-Filter gefahren (SAFE)."""
    filters = [PeakingFilter("peq", 100.0, -4.0, 1.0)] if filters is None else filters
    nh = _real_no_harm(filters, filters) if no_harm is None else no_harm
    sat = [EqProposal(f, "mode") for f in filters]
    args = dict(run_id="r", target_device="kef_ls50_wireless", no_harm=nh, measurement_set_id="m",
                eq_proposals_l=sat, eq_proposals_r=sat)
    args.update(kw)
    return build_filter_export(**args)


def _legacy_from(doc):
    """Baut ein gueltiges Legacy-1.0-Dokument (eigener 1.0-Digest, RELEASED) aus einem 1.1-Dokument."""
    legacy = copy.deepcopy(doc)
    legacy["contract_version"] = "1.0"
    legacy["approval"] = {
        "status": "RELEASED", "do_no_harm": "SAFE", "hardware_validated": False,
        "checked_filters_sha256": compute_filters_digest(legacy["filters"]),
    }
    return legacy


# --- Aufbau und Grundfunktion ----------------------------------------------------
def test_saxx_export_is_always_blocked_with_no_scope_even_on_safe():
    """Regression: Saxx-Exporte duerfen nie RELEASED werden (kein Subwoofer-Do-No-Harm-Modell)."""
    doc = _sub_doc()
    assert doc["approval"] == {
        "status": "BLOCKED", "do_no_harm": "SAFE", "hardware_validated": False,
        "checked_scope": "none", "checked_filters_sha256": None,
    }
    assert doc["contract_version"] == "1.1"
    assert doc["filters"]["eq_proposals_l"] == [] and len(doc["filters"]["subwoofer"]) == 1
    schema = json.loads(SCHEMA.read_text("utf-8"))
    assert set(schema["required"]) == set(doc)
    assert schema["properties"]["contract_version"]["const"] == doc["contract_version"] == "1.1"
    assert set(schema["properties"]["approval"]["required"]) == set(doc["approval"])


@pytest.mark.parametrize("status", [NoHarmStatus.UNSAFE, NoHarmStatus.INCONCLUSIVE])
def test_unreleased_status_is_blocked(status):
    doc = _sub_doc(no_harm=NoHarmResult(status, "x", 1))
    assert doc["approval"]["status"] == "BLOCKED" and doc["approval"]["do_no_harm"] == status.value


def test_satellite_filters_never_exported_to_subwoofer_device():
    sat = [EqProposal(PeakingFilter("peq", 80.0, -2.0, 3.0), "mode")]
    with pytest.raises(FilterExportError, match="Satellitenfilter"):
        _sub_doc(eq_proposals_l=sat)
    doc = _sub_doc()
    doc["filters"]["eq_proposals_r"] = [{"type": "peq", "fc_hz": 80.0, "gain_db": -2.0, "q": 3.0}]
    with pytest.raises(FilterExportError, match="Satellitenfilter"):
        validate_filter_export(doc)


def test_kef_target_keeps_lists_separate_and_rejects_sub_filters():
    sat = [EqProposal(PeakingFilter("peq", 80.0, -2.0, 3.0), "mode")]
    doc = build_filter_export(run_id="r", target_device="kef_ls50_wireless", no_harm=SAFE,
                              measurement_set_id="m", eq_proposals_l=sat)
    assert len(doc["filters"]["eq_proposals_l"]) == 1 and doc["filters"]["subwoofer"] == []
    with pytest.raises(FilterExportError, match="Subwoofer-Filter"):
        build_filter_export(run_id="r", target_device="kef_ls50_wireless", no_harm=SAFE, measurement_set_id="m",
                            subwoofer_filters=[PeakingFilter("peq", 50.0, -3.0, 4.0)])


def test_checked_subwoofer_filters_parameter_is_removed():
    """Regression: Subwoofer-Freigabe per Selbstauskunft (checked_subwoofer_filters) ist entfernt."""
    with pytest.raises(TypeError):
        build_filter_export(run_id="r", target_device="saxx_ds120", no_harm=SAFE, measurement_set_id="m",
                            subwoofer_filters=[PeakingFilter("peq", 50.0, -3.0, 4.0)],
                            checked_subwoofer_filters=[PeakingFilter("peq", 50.0, -3.0, 4.0)])


def test_empty_export_needs_intentional_flat():
    with pytest.raises(FilterExportError, match="intentional_flat"):
        _sub_doc(subwoofer_filters=[])
    assert _sub_doc(subwoofer_filters=[], intentional_flat=True)["intentional_flat"] is True
    with pytest.raises(FilterExportError, match="widerspricht"):
        _sub_doc(intentional_flat=True)


def test_hardware_validated_true_and_inconsistent_release_rejected():
    doc = _sub_doc()
    bad = copy.deepcopy(doc)
    bad["approval"]["hardware_validated"] = True
    with pytest.raises(FilterExportError, match="hardware_validated"):
        validate_filter_export(bad)
    bad = copy.deepcopy(_kef_doc())
    bad["approval"]["do_no_harm"] = "UNSAFE"  # RELEASED bleibt gesetzt
    with pytest.raises(FilterExportError, match="RELEASED"):
        validate_filter_export(bad)


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("run_id"),
    lambda d: d.update(extra=1),
    lambda d: d["approval"].update(extra=1),
    lambda d: d.update(contract_version="2.0"),
    lambda d: d.update(contract_version="1.2"),
    lambda d: d["filters"]["subwoofer"][0].update(fc_hz=float("nan")),
    lambda d: d["filters"]["subwoofer"][0].update(q=0),
    lambda d: d["filters"]["subwoofer"][0].update(type="lowpass"),
    lambda d: d["filters"]["subwoofer"][0].update(shelf_slope_db=6),
    lambda d: d["provenance"].update(source_profile_sha256="ABC"),
    lambda d: d.update(target_device="unknown"),
    lambda d: d["approval"].pop("checked_scope"),
    lambda d: d["approval"].update(checked_scope="all"),
    lambda d: d["approval"].update(checked_scope="none", checked_filters_sha256="a" * 64),
    lambda d: d["approval"].update(checked_scope="left_right", checked_filters_sha256=None),
])
def test_invalid_documents_rejected(mutate):
    doc = _sub_doc()
    mutate(doc)
    with pytest.raises(FilterExportError):
        validate_filter_export(doc)


def test_non_dict_rejected():
    with pytest.raises(FilterExportError):
        validate_filter_export([])


def test_unsupported_filter_type_not_built():
    with pytest.raises(FilterExportError, match="nicht exportierbar"):
        _sub_doc(subwoofer_filters=[PeakingFilter("lowpass", 100.0)])


def test_deterministic_write_roundtrip_no_overwrite_and_source_untouched(tmp_path):
    filters = [PeakingFilter("peq", 50.0, -3.0, 4.0)]
    doc = _sub_doc(subwoofer_filters=filters)
    assert dumps_filter_export(doc) == dumps_filter_export(copy.deepcopy(doc))
    p = write_filter_export(tmp_path / "filter_export.json", doc)
    assert load_filter_export(p) == doc
    with pytest.raises(FilterExportError, match="Ueberschreiben"):
        write_filter_export(p, doc)
    write_filter_export(p, doc, overwrite=True)
    assert filters == [PeakingFilter("peq", 50.0, -3.0, 4.0)]
    p.write_text("{kaputt", encoding="utf-8")
    with pytest.raises(FilterExportError, match="nicht lesbar"):
        load_filter_export(p)


def test_kef_roundtrip_is_deterministic_and_released(tmp_path):
    doc = _kef_doc()
    p = write_filter_export(tmp_path / "kef.json", doc)
    again = write_filter_export(tmp_path / "kef2.json", copy.deepcopy(doc))
    assert p.read_text("utf-8") == again.read_text("utf-8")
    loaded = load_filter_export(p)
    assert loaded == doc and is_device_released(loaded) is True


def test_invalid_doc_not_written(tmp_path):
    doc = _sub_doc()
    doc["run_id"] = ""
    with pytest.raises(FilterExportError):
        write_filter_export(tmp_path / "x.json", doc)
    assert not (tmp_path / "x.json").exists()


# --- Typsicherheit, SHA-fullmatch, Tiefenkopie ---------------------------------------
@pytest.mark.parametrize("path,value", [
    (("approval", "do_no_harm"), ["SAFE"]), (("approval", "do_no_harm"), {"a": 1}), (("approval", "do_no_harm"), 5),
    (("approval", "status"), ["RELEASED"]), (("approval", "status"), {}), (("target_device",), ["saxx_ds120"]),
    (("target_device",), {"x": 1}), (("run_id",), ["r"]), (("contract_version",), ["1.1"]),
    (("provenance", "source_profile_sha256"), ["a" * 64]), (("approval", "checked_filters_sha256"), ["a" * 64]),
    (("approval", "checked_filters_sha256"), {"a": 1}), (("approval", "checked_scope"), ["none"]),
    (("filters", "subwoofer", 0, "type"), ["peq"]), (("filters", "subwoofer", 0, "type"), {"a": 1}),
    (("filters", "subwoofer", 0, "fc_hz"), 10 ** 400),
    (("filters", "subwoofer", 0, "shelf_slope_db"), [6]),
])
def test_unhashable_or_huge_values_raise_filter_export_error_not_typeerror(path, value):
    doc = _sub_doc()
    o = doc
    for k in path[:-1]:
        o = o[k]
    o[path[-1]] = value
    with pytest.raises(FilterExportError):
        validate_filter_export(doc)


@pytest.mark.parametrize("bad", ["a" * 64 + "\n", "\n" + "a" * 64, "A" * 64, "a" * 63, "a" * 65, " " + "a" * 64])
def test_sha256_fields_use_fullmatch(bad):
    for where in (("provenance", "source_profile_sha256"), ("approval", "checked_filters_sha256")):
        doc = _sub_doc()
        doc[where[0]][where[1]] = bad
        with pytest.raises(FilterExportError):
            validate_filter_export(doc)


def test_crossovers_are_deep_copied_both_ways():
    xo = {"hpf": {"frequency_hz": 25.0, "type": "24db-LR", "enabled": True}, "lpf": None, "phase_deg": 90}
    snapshot = copy.deepcopy(xo)
    doc = _sub_doc(crossovers=xo)
    doc["crossovers"]["hpf"]["enabled"] = False  # Export aendern ...
    assert xo == snapshot  # ... laesst die Eingabe unberuehrt
    xo["hpf"]["frequency_hz"] = 99.0  # Eingabe spaeter aendern ...
    assert doc["crossovers"]["hpf"]["frequency_hz"] == 25.0  # ... laesst den Export unberuehrt


# --- Freigabebindung KEF (Scope und Digest aus validate_no_harm) ----------------------
def test_kef_real_safe_check_is_released_with_left_right_scope():
    doc = _kef_doc()
    assert doc["approval"]["status"] == "RELEASED"
    assert doc["approval"]["checked_scope"] == CHECKED_SCOPE_LEFT_RIGHT
    assert doc["approval"]["checked_filters_sha256"] == filters_digest(doc["filters"])
    assert is_device_released(doc) is True


def test_kef_hand_made_safe_without_binding_is_blocked():
    """Regression: ein selbst gebautes SAFE ohne Scope/Digest gibt KEF keine Freigabe mehr."""
    doc = build_filter_export(
        run_id="r", target_device="kef_ls50_wireless", no_harm=NoHarmResult(NoHarmStatus.SAFE, "x", 5),
        measurement_set_id="m", eq_proposals_l=[EqProposal(PeakingFilter("peq", 80.0, -2.0, 3.0), "mode")],
    )
    assert doc["approval"]["status"] == "BLOCKED"
    assert doc["approval"]["checked_scope"] == "none" and doc["approval"]["checked_filters_sha256"] is None
    assert is_device_released(doc) is False


def _forge(result, **attrs):
    """BEWUSSTE FAELSCHUNG (nur Test): setzt gesperrte Felder an einem fertigen Ergebnis vorbei."""
    for k, v in attrs.items():
        object.__setattr__(result, k, v)
    return result


def test_kef_safe_result_with_foreign_digest_is_blocked():
    """Regression: Pruefungsnachweis fuer ANDERE Filter (fremder Digest) gibt keine Freigabe.
    Der Nachweis wird bewusst per object.__setattr__ gefaelscht (Konstruktor verbietet das)."""
    other = [PeakingFilter("peq", 60.0, -6.0, 2.0)]
    foreign = _forge(
        NoHarmResult(NoHarmStatus.SAFE, "x", 5),
        checked_scope=CHECKED_SCOPE_LEFT_RIGHT, checked_filters_sha256=peaking_filters_digest(other, other),
    )
    doc = _kef_doc(no_harm=foreign)
    assert doc["approval"]["status"] == "BLOCKED"
    assert doc["approval"]["checked_filters_sha256"] == peaking_filters_digest(other, other)


def test_constructor_cannot_fake_check_in_kef_export():
    """Regression N1: NoHarmResult(SAFE, 'fake', ...) mit Pruefnachweis ist per Konstruktor nicht moeglich."""
    pf = PeakingFilter("peq", 80.0, -2.0, 3.0)
    with pytest.raises(TypeError):
        NoHarmResult(NoHarmStatus.SAFE, "fake", 0, checked_scope="left_right",
                     checked_filters_sha256=peaking_filters_digest([pf], []))
    fake = NoHarmResult(NoHarmStatus.SAFE, "fake", 5)  # ohne Nachweis
    doc = build_filter_export(run_id="r", target_device="kef_ls50_wireless", no_harm=fake,
                              measurement_set_id="m", eq_proposals_l=[EqProposal(pf, "mode")])
    assert doc["approval"]["status"] == "BLOCKED"


def test_forged_too_few_positions_blocks_kef_release():
    """DELIBERATE FORGERY (Test): gueltiger Nachweis, aber n_positions unter der Mindestzahl -> BLOCKED."""
    f = [PeakingFilter("peq", 100.0, -4.0, 1.0)]
    real = _real_no_harm(f, f)
    forged = _forge(NoHarmResult(NoHarmStatus.SAFE, "x", 2),
                    checked_scope=real.checked_scope, checked_filters_sha256=real.checked_filters_sha256)
    sat = [EqProposal(p, "mode") for p in f]
    doc = build_filter_export(run_id="r", target_device="kef_ls50_wireless", no_harm=forged,
                              measurement_set_id="m", eq_proposals_l=sat, eq_proposals_r=sat)
    assert doc["approval"]["status"] == "BLOCKED"


def test_forged_left_right_without_digest_raises_filter_export_error():
    """DELIBERATE FORGERY (Test): inkonsistentes Paar left_right + None -> fail-closed Fehler, nie RELEASED."""
    f = [PeakingFilter("peq", 100.0, -4.0, 1.0)]
    forged = _forge(NoHarmResult(NoHarmStatus.SAFE, "x", 5), checked_scope=CHECKED_SCOPE_LEFT_RIGHT)
    sat = [EqProposal(p, "mode") for p in f]
    with pytest.raises(FilterExportError):
        build_filter_export(run_id="r", target_device="kef_ls50_wireless", no_harm=forged,
                            measurement_set_id="m", eq_proposals_l=sat, eq_proposals_r=sat)


# --- N3: Schema-if/then-Regeln (Schema UND Python lehnen ab) ------------------------------
def test_schema_if_then_rules_reject_scope_digest_and_release_violations():
    v = _schema_validator()
    cases = []
    d = _sub_doc()  # (i) scope none mit Digest
    d["approval"]["checked_filters_sha256"] = "a" * 64
    cases.append(d)
    d = _kef_doc()  # (ii) left_right ohne Digest
    d["approval"]["checked_filters_sha256"] = None
    cases.append(d)
    d = _sub_doc()  # (iii) RELEASED fuer Saxx
    d["approval"]["status"] = "RELEASED"
    cases.append(d)
    d = _kef_doc()  # (iii) RELEASED mit do_no_harm != SAFE
    d["approval"]["do_no_harm"] = "UNSAFE"
    cases.append(d)
    d = _kef_doc()  # (iii) RELEASED mit scope none
    d["approval"]["checked_scope"] = "none"
    d["approval"]["checked_filters_sha256"] = None
    cases.append(d)
    for doc in cases:
        assert not v.is_valid(doc), json.dumps(doc["approval"])
        with pytest.raises(FilterExportError):
            validate_filter_export(doc)


def test_kef_real_release_is_accepted_by_schema_if_then_rules():
    v = _schema_validator()
    assert v.is_valid(_kef_doc())


def test_kef_exported_filters_changed_after_check_are_blocked_and_manual_release_fails():
    checked = [PeakingFilter("peq", 100.0, -4.0, 1.0)]
    nh = _real_no_harm(checked, checked)
    assert nh.checked_scope == CHECKED_SCOPE_LEFT_RIGHT
    changed = [EqProposal(PeakingFilter("peq", 100.0, -8.0, 1.0), "mode")]  # Gain nach der Pruefung geaendert
    doc = build_filter_export(run_id="r", target_device="kef_ls50_wireless", no_harm=nh,
                              measurement_set_id="m", eq_proposals_l=changed, eq_proposals_r=changed)
    assert doc["approval"]["status"] == "BLOCKED"
    doc["approval"]["status"] = "RELEASED"  # manuell auf RELEASED gesetzt
    with pytest.raises(FilterExportError, match="checked_filters_sha256"):
        validate_filter_export(doc)


def test_kef_negative_zero_gain_still_released_g3():
    """Regression G3: -0.0 und 0.0 im exportierten Gain sind dieselbe Freigabe (kein Fehlalarm)."""
    zero = [PeakingFilter("peq", 100.0, 0.0, 1.0)]
    nh = _real_no_harm(zero, zero)
    assert nh.status is NoHarmStatus.SAFE
    sat = [EqProposal(PeakingFilter("peq", 100.0, -0.0, 1.0), "mode")]
    doc = build_filter_export(run_id="r", target_device="kef_ls50_wireless", no_harm=nh,
                              measurement_set_id="m", eq_proposals_l=sat, eq_proposals_r=sat)
    assert doc["approval"]["status"] == "RELEASED"


def test_kef_no_filters_with_safe_no_filter_result_is_blocked():
    nh = _real_no_harm([], [])
    assert nh.status is NoHarmStatus.SAFE and nh.checked_scope == CHECKED_SCOPE_NONE
    doc = build_filter_export(run_id="r", target_device="kef_ls50_wireless", no_harm=nh,
                              measurement_set_id="m", intentional_flat=True)
    assert doc["approval"]["status"] == "BLOCKED" and doc["approval"]["checked_scope"] == "none"
    assert is_device_released(doc) is False


@pytest.mark.parametrize("field,val", [("fc_hz", 60.0), ("gain_db", -9.0), ("q", 8.0)])
def test_filters_changed_after_kef_check_invalidate_released_export(field, val):
    doc = _kef_doc()
    assert doc["approval"]["status"] == "RELEASED"
    doc["filters"]["eq_proposals_l"][0][field] = val
    with pytest.raises(FilterExportError, match="checked_filters_sha256"):
        validate_filter_export(doc)
    doc["approval"]["status"] = "BLOCKED"  # als nicht freigegeben bleibt er gueltig
    validate_filter_export(doc)


def test_added_filter_after_kef_check_invalidates_release():
    doc = _kef_doc()
    doc["filters"]["eq_proposals_r"].append({"type": "peq", "fc_hz": 90.0, "gain_db": -1.0, "q": 2.0})
    with pytest.raises(FilterExportError, match="checked_filters_sha256"):
        validate_filter_export(doc)


def test_kef_scope_none_with_digest_or_left_right_without_digest_rejected():
    doc = _kef_doc()
    doc["approval"]["checked_scope"] = "none"  # sha bleibt gesetzt
    with pytest.raises(FilterExportError, match="checked_scope 'none'"):
        validate_filter_export(doc)
    doc = _kef_doc()
    doc["approval"]["checked_filters_sha256"] = None  # left_right ohne sha
    with pytest.raises(FilterExportError, match="left_right"):
        validate_filter_export(doc)


def test_digest_is_integrity_not_authenticity_known_limit():
    """Dokumentiert die bekannte Grenze: wer Filter UND Digest gezielt neu schreibt,
    erzeugt ein gueltiges Dokument (kein Geheimnis/keine Signatur)."""
    doc = _kef_doc()
    doc["filters"]["eq_proposals_l"][0]["gain_db"] = -8.0
    doc["approval"]["checked_filters_sha256"] = filters_digest(doc["filters"])
    assert validate_filter_export(doc) is doc


# --- Saxx und Mosconi: nie freigegeben ------------------------------------------------
def test_saxx_with_real_safe_left_right_result_is_blocked():
    """Regression: auch eine echte SAFE-Pruefung ueber links/rechts gibt Saxx keine Freigabe."""
    f = [PeakingFilter("peq", 100.0, -4.0, 1.0)]
    doc = build_filter_export(run_id="r", target_device="saxx_ds120", no_harm=_real_no_harm(f, f),
                              measurement_set_id="m", subwoofer_filters=f)
    assert doc["approval"]["status"] == "BLOCKED"
    assert doc["approval"]["checked_scope"] == "none" and doc["approval"]["checked_filters_sha256"] is None


def test_mosconi_always_blocked_even_with_real_safe_result():
    f = [PeakingFilter("peq", 100.0, -4.0, 1.0)]
    doc = build_filter_export(run_id="r", target_device="mosconi_pico_8_10", no_harm=_real_no_harm(f, f),
                              measurement_set_id="m", subwoofer_filters=f)
    assert doc["approval"]["status"] == "BLOCKED"
    assert doc["approval"]["checked_scope"] == "none" and doc["approval"]["checked_filters_sha256"] is None
    assert is_device_released(doc) is False


@pytest.mark.parametrize("device", ["saxx_ds120", "mosconi_pico_8_10"])
def test_manually_released_subwoofer_export_fails_validation(device):
    doc = _sub_doc(target_device=device)
    doc["approval"]["status"] = "RELEASED"
    with pytest.raises(FilterExportError, match="Saxx/Mosconi-Exporte sind ohne eigenes Do-No-Harm-Modell"):
        validate_filter_export(doc)


def test_saxx_manually_released_with_left_right_scope_fails_validation():
    doc = _sub_doc()
    doc["approval"].update(status="RELEASED", checked_scope="left_right", checked_filters_sha256=compute_filters_digest(doc["filters"]))
    with pytest.raises(FilterExportError, match="Saxx/Mosconi"):
        validate_filter_export(doc)


def test_released_still_requires_safe_and_inconclusive_is_blocked():
    doc = _sub_doc(no_harm=NoHarmResult(NoHarmStatus.INCONCLUSIVE, "x", 1))
    assert doc["approval"]["status"] == "BLOCKED"
    doc["approval"]["status"] = "RELEASED"
    with pytest.raises(FilterExportError, match="nur bei do_no_harm SAFE"):
        validate_filter_export(doc)


# --- Legacy 1.0: nur lesend -------------------------------------------------------------
def test_legacy_1_0_document_validates_but_never_grants_device_release(tmp_path):
    legacy = _legacy_from(_sub_doc())
    assert validate_filter_export(legacy) is legacy
    assert is_device_released(legacy) is False
    with pytest.raises(AdapterBlockedError):
        map_export_to_edit(legacy)
    assert build_preview(legacy).released is False
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_filter_export(p) == legacy


def test_legacy_1_0_kef_release_is_not_device_release():
    legacy = _legacy_from(_kef_doc())
    assert validate_filter_export(legacy) is legacy
    assert is_device_released(legacy) is False


def test_invalid_legacy_1_0_documents_still_rejected():
    legacy = _legacy_from(_sub_doc())
    legacy["filters"]["subwoofer"][0]["gain_db"] = -9.0  # Digest passt nicht mehr
    with pytest.raises(FilterExportError, match="checked_filters_sha256"):
        validate_filter_export(legacy)
    legacy = _legacy_from(_sub_doc())
    legacy["approval"]["hardware_validated"] = True
    with pytest.raises(FilterExportError, match="hardware_validated"):
        validate_filter_export(legacy)


def test_legacy_1_0_is_never_rewritten():
    legacy = _legacy_from(_sub_doc())
    with pytest.raises(FilterExportError, match="Legacy-1.0-Exporte sind nur lesbar"):
        dumps_filter_export(legacy)
    with pytest.raises(FilterExportError, match="Legacy-1.0-Exporte sind nur lesbar"):
        write_filter_export("unused.json", legacy)


def test_legacy_1_0_schema_file_accepts_valid_legacy_document():
    jsonschema = pytest.importorskip("jsonschema")
    v = jsonschema.Draft202012Validator(json.loads(SCHEMA_10.read_text("utf-8")))
    legacy = _legacy_from(_sub_doc())
    assert v.is_valid(legacy)
    legacy["approval"]["hardware_validated"] = True
    assert not v.is_valid(legacy)


# --- Digest-Regeln (1.1, filter_digest) -----------------------------------------------
def test_digest_covers_values_order_and_list_membership():
    f1, f2 = PeakingFilter("peq", 50.0, -3.0, 4.0), PeakingFilter("peq", 70.0, -2.0, 3.0)
    d12 = filters_digest(_sub_doc(subwoofer_filters=[f1, f2])["filters"])
    assert d12 != filters_digest(_sub_doc(subwoofer_filters=[f2, f1])["filters"])  # Reihenfolge
    assert d12 != filters_digest(_sub_doc(subwoofer_filters=[f1])["filters"])
    moved = _sub_doc(subwoofer_filters=[f1, f2])["filters"]
    moved["eq_proposals_l"], moved["subwoofer"] = moved["subwoofer"], []  # gleiche Filter, andere Liste
    assert d12 != filters_digest(moved)
    for field, val in (("fc_hz", 51.0), ("gain_db", -3.5), ("q", 4.5), ("type", "lowshelf")):
        changed = copy.deepcopy(_sub_doc(subwoofer_filters=[f1, f2])["filters"])
        changed["subwoofer"][0][field] = val
        assert filters_digest(changed) != d12, field
    ints = copy.deepcopy(_sub_doc(subwoofer_filters=[f1, f2])["filters"])
    ints["subwoofer"][0]["fc_hz"] = 50  # int statt float -> gleicher Digest
    assert filters_digest(ints) == d12


def test_legacy_compute_filters_digest_rejects_malformed_input():
    for bad in ([], {"eq_proposals_l": [], "eq_proposals_r": []},
                {"eq_proposals_l": [], "eq_proposals_r": [], "subwoofer": [{"type": "peq"}]},
                {"eq_proposals_l": [], "eq_proposals_r": [], "subwoofer": [{"type": "peq", "fc_hz": "x", "gain_db": 0, "q": 1}]}):
        with pytest.raises(FilterExportError):
            compute_filters_digest(bad)


# --- JSON-Schema-Konformitaet (1.1 vs. Python-Validierung) ------------------------------
def _schema_validator():
    jsonschema = pytest.importorskip("jsonschema")  # optionale Dev-Abhaengigkeit; ueberspringt nur diese Tests
    schema = json.loads(SCHEMA.read_text("utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _valid_corpus():
    sub = [PeakingFilter("peq", 50.0, -3.0, 4.0), PeakingFilter("lowshelf", 60.0, 2.0, 0.7)]
    xo = {"hpf": {"frequency_hz": 25.0, "type": "24db-LR", "enabled": True}, "lpf": None, "phase_deg": 90}
    return [
        _sub_doc(),
        _sub_doc(subwoofer_filters=sub, subwoofer_shelf_slopes_db={1: 12}, crossovers=xo, source_profile_sha256="b" * 64),
        _sub_doc(subwoofer_filters=[], intentional_flat=True),
        _sub_doc(no_harm=NoHarmResult(NoHarmStatus.UNSAFE, "x", 3)),
        _sub_doc(no_harm=NoHarmResult(NoHarmStatus.INCONCLUSIVE, "x", 0)),
        _sub_doc(target_device="mosconi_pico_8_10", crossovers=xo),
        _kef_doc(),
        _kef_doc(no_harm=NoHarmResult(NoHarmStatus.UNSAFE, "x", 3)),
        _kef_doc(filters=[PeakingFilter("peq", 100.0, -4.0, 1.0), PeakingFilter("peq", 300.0, -2.0, 3.0)],
                 crossovers=xo),
    ]


def test_every_valid_export_conforms_to_published_schema():
    v = _schema_validator()
    for doc in _valid_corpus():
        validate_filter_export(doc)
        errors = list(v.iter_errors(doc))
        assert not errors, [e.message for e in errors]
        v.validate(json.loads(dumps_filter_export(doc)))  # auch nach dem Serialisieren


def test_python_accepts_implies_schema_accepts_under_random_mutation():
    """Differenzialtest (deterministisch): Python-valid => Schema-valid."""
    import random
    v = _schema_validator()
    junk = [None, True, False, 0, 1, -1, 1.5, "", "x", [], {}, "a" * 64, "a" * 64 + "\n", 6, 12, 6.0, ["SAFE"], "none", "left_right"]

    def paths(o, pre=()):
        yield pre
        if isinstance(o, dict):
            for k, val in o.items():
                yield from paths(val, pre + (k,))
        elif isinstance(o, list):
            for i, val in enumerate(o):
                yield from paths(val, pre + (i,))

    rng = random.Random(20261126)
    corpus, accepted = _valid_corpus(), 0
    for _ in range(2500):
        d = copy.deepcopy(rng.choice(corpus))
        for _ in range(rng.randint(1, 3)):
            p = rng.choice([p for p in paths(d) if p])
            o = d
            for k in p[:-1]:
                o = o[k]
            if rng.random() < 0.15:
                del o[p[-1]]
            else:
                o[p[-1]] = rng.choice(junk)
        try:
            validate_filter_export(d)  # darf nie etwas anderes als FilterExportError werfen
        except FilterExportError:
            continue
        accepted += 1
        assert v.is_valid(d), json.dumps(d)[:400]
    assert accepted > 0


@pytest.mark.parametrize("mutate", [
    lambda d: d["approval"].pop("checked_filters_sha256"),
    lambda d: d["approval"].pop("checked_scope"),
    lambda d: d["approval"].update(checked_filters_sha256="A" * 64),
    lambda d: d["approval"].update(checked_scope="all"),
    lambda d: d["approval"].update(hardware_validated=True),
    lambda d: d.update(extra=1),
    lambda d: d["crossovers"].update(foo=1),
    lambda d: d["provenance"].update(source_profile_sha256="x"),
])
def test_schema_and_python_both_reject_structural_errors(mutate):
    d = _sub_doc()
    mutate(d)
    assert not _schema_validator().is_valid(d)
    with pytest.raises(FilterExportError):
        validate_filter_export(d)
