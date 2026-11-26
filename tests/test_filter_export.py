"""Tests fuer filter_export.py (synthetische Daten)."""

import copy
import json
from pathlib import Path

import pytest

from spatial_analyzer.biquad_preview import PeakingFilter
from spatial_analyzer.eq_solver import EqProposal, NoHarmResult, NoHarmStatus
from spatial_analyzer.filter_export import (
    FilterExportError,
    build_filter_export,
    compute_filters_digest,
    dumps_filter_export,
    load_filter_export,
    validate_filter_export,
    write_filter_export,
)

SAFE = NoHarmResult(NoHarmStatus.SAFE, "ok", 5)
SCHEMA = Path(__file__).resolve().parents[1] / "spatial_analyzer" / "adapters" / "schemas" / "filter_export.schema.json"


def _sub_doc(**kw):
    args = dict(
        run_id="r1", target_device="saxx_ds120", no_harm=SAFE, measurement_set_id="m1",
        subwoofer_filters=[PeakingFilter("peq", 50.0, -3.0, 4.0)],
    )
    args.update(kw)
    args.setdefault("checked_subwoofer_filters", args["subwoofer_filters"])  # Pruefung deckt die Filter ab
    return build_filter_export(**args)


def test_build_valid_sub_export_and_schema_file_matches_keys():
    doc = _sub_doc()
    assert doc["approval"] == {
        "status": "RELEASED", "do_no_harm": "SAFE", "hardware_validated": False,
        "checked_filters_sha256": compute_filters_digest(doc["filters"]),
    }
    assert doc["filters"]["eq_proposals_l"] == [] and len(doc["filters"]["subwoofer"]) == 1
    schema = json.loads(SCHEMA.read_text("utf-8"))
    assert set(schema["required"]) == set(doc)
    assert schema["properties"]["contract_version"]["const"] == doc["contract_version"] == "1.0"


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


def test_empty_export_needs_intentional_flat():
    with pytest.raises(FilterExportError, match="intentional_flat"):
        _sub_doc(subwoofer_filters=[])
    assert _sub_doc(subwoofer_filters=[], intentional_flat=True)["intentional_flat"] is True
    with pytest.raises(FilterExportError, match="widerspricht"):
        _sub_doc(intentional_flat=True)


def test_hardware_validated_true_and_inconsistent_approval_rejected():
    doc = _sub_doc()
    bad = copy.deepcopy(doc)
    bad["approval"]["hardware_validated"] = True
    with pytest.raises(FilterExportError, match="hardware_validated"):
        validate_filter_export(bad)
    bad = copy.deepcopy(doc)
    bad["approval"]["do_no_harm"] = "UNSAFE"
    with pytest.raises(FilterExportError, match="RELEASED"):
        validate_filter_export(bad)


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("run_id"),
    lambda d: d.update(extra=1),
    lambda d: d.update(contract_version="2.0"),
    lambda d: d["filters"]["subwoofer"][0].update(fc_hz=float("nan")),
    lambda d: d["filters"]["subwoofer"][0].update(q=0),
    lambda d: d["filters"]["subwoofer"][0].update(type="lowpass"),
    lambda d: d["filters"]["subwoofer"][0].update(shelf_slope_db=6),
    lambda d: d["provenance"].update(source_profile_sha256="ABC"),
    lambda d: d.update(target_device="unknown"),
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


def test_invalid_doc_not_written(tmp_path):
    doc = _sub_doc()
    doc["run_id"] = ""
    with pytest.raises(FilterExportError):
        write_filter_export(tmp_path / "x.json", doc)
    assert not (tmp_path / "x.json").exists()


# --- Korrekturrunde: Typsicherheit, SHA-fullmatch, Tiefenkopie ---------------------
@pytest.mark.parametrize("path,value", [
    (("approval", "do_no_harm"), ["SAFE"]), (("approval", "do_no_harm"), {"a": 1}), (("approval", "do_no_harm"), 5),
    (("approval", "status"), ["RELEASED"]), (("approval", "status"), {}), (("target_device",), ["saxx_ds120"]),
    (("target_device",), {"x": 1}), (("run_id",), ["r"]), (("contract_version",), ["1.0"]),
    (("provenance", "source_profile_sha256"), ["a" * 64]), (("approval", "checked_filters_sha256"), ["a" * 64]),
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


# --- Freigabebindung an die Filter (W1/W2) -------------------------------------------
def test_subwoofer_filters_not_covered_by_check_stay_blocked():
    doc = _sub_doc(checked_subwoofer_filters=None)  # z. B. SAFE der Pipeline gilt nur fuer die Satelliten
    assert doc["approval"]["do_no_harm"] == "SAFE" and doc["approval"]["status"] == "BLOCKED"
    other = [PeakingFilter("peq", 50.0, -6.0, 4.0)]  # geprueft wurde ein ANDERER Filter
    assert _sub_doc(checked_subwoofer_filters=other)["approval"]["status"] == "BLOCKED"


def test_covered_filters_are_released_and_reason_or_slope_do_not_matter():
    f = [PeakingFilter("lowshelf", 60.0, 2.0, 0.7)]
    a = _sub_doc(subwoofer_filters=f, checked_subwoofer_filters=f, subwoofer_shelf_slopes_db={0: 6})
    b = _sub_doc(subwoofer_filters=f, checked_subwoofer_filters=f, subwoofer_shelf_slopes_db={0: 12})
    assert a["approval"]["status"] == "RELEASED"
    assert a["approval"]["checked_filters_sha256"] == b["approval"]["checked_filters_sha256"]


def test_digest_covers_values_order_and_list_membership():
    f1, f2 = PeakingFilter("peq", 50.0, -3.0, 4.0), PeakingFilter("peq", 70.0, -2.0, 3.0)
    d12 = compute_filters_digest(_sub_doc(subwoofer_filters=[f1, f2])["filters"])
    assert d12 != compute_filters_digest(_sub_doc(subwoofer_filters=[f2, f1])["filters"])  # Reihenfolge
    assert d12 != compute_filters_digest(_sub_doc(subwoofer_filters=[f1])["filters"])
    moved = _sub_doc(subwoofer_filters=[f1, f2])["filters"]
    moved["eq_proposals_l"], moved["subwoofer"] = moved["subwoofer"], []  # gleiche Filter, andere Liste
    assert d12 != compute_filters_digest(moved)
    for field, val in (("fc_hz", 51.0), ("gain_db", -3.5), ("q", 4.5), ("type", "lowshelf")):
        changed = copy.deepcopy(_sub_doc(subwoofer_filters=[f1, f2])["filters"])
        changed["subwoofer"][0][field] = val
        assert compute_filters_digest(changed) != d12, field
    ints = copy.deepcopy(_sub_doc(subwoofer_filters=[f1, f2])["filters"])
    ints["subwoofer"][0]["fc_hz"] = 50  # int statt float -> gleicher Digest
    assert compute_filters_digest(ints) == d12


@pytest.mark.parametrize("field,val", [("fc_hz", 60.0), ("gain_db", -9.0), ("q", 8.0)])
def test_filters_changed_after_check_invalidate_released_export(field, val):
    doc = _sub_doc()
    assert doc["approval"]["status"] == "RELEASED"
    doc["filters"]["subwoofer"][0][field] = val
    with pytest.raises(FilterExportError, match="checked_filters_sha256"):
        validate_filter_export(doc)
    doc["approval"]["status"] = "BLOCKED"  # als nicht freigegeben bleibt er gueltig
    validate_filter_export(doc)


def test_added_filter_after_check_invalidates_release():
    doc = _sub_doc()
    doc["filters"]["subwoofer"].append({"type": "peq", "fc_hz": 90.0, "gain_db": -1.0, "q": 2.0})
    with pytest.raises(FilterExportError, match="checked_filters_sha256"):
        validate_filter_export(doc)


def test_released_still_requires_safe_and_inconclusive_with_coverage_is_blocked():
    doc = _sub_doc(no_harm=NoHarmResult(NoHarmStatus.INCONCLUSIVE, "x", 1))
    assert doc["approval"]["status"] == "BLOCKED"
    doc["approval"]["status"] = "RELEASED"
    with pytest.raises(FilterExportError, match="nur bei do_no_harm SAFE"):
        validate_filter_export(doc)


def test_kef_export_with_satellite_check_is_released_and_binds_satellites():
    sat = [EqProposal(PeakingFilter("peq", 80.0, -2.0, 3.0), "mode")]
    doc = build_filter_export(run_id="r", target_device="kef_ls50_wireless", no_harm=SAFE,
                              measurement_set_id="m", eq_proposals_l=sat)
    assert doc["approval"]["status"] == "RELEASED"
    doc["filters"]["eq_proposals_l"][0]["gain_db"] = -8.0
    with pytest.raises(FilterExportError, match="checked_filters_sha256"):
        validate_filter_export(doc)


def test_digest_is_integrity_not_authenticity_known_limit():
    """Dokumentiert die bekannte Grenze: wer Filter UND Digest UND Status gezielt
    neu schreibt, erzeugt ein gueltiges Dokument (kein Geheimnis/keine Signatur)."""
    doc = _sub_doc(no_harm=NoHarmResult(NoHarmStatus.UNSAFE, "x", 1))
    assert doc["approval"]["status"] == "BLOCKED"
    doc["approval"].update(status="RELEASED", do_no_harm="SAFE")
    assert validate_filter_export(doc) is doc


def test_compute_filters_digest_rejects_malformed_input():
    for bad in ([], {"eq_proposals_l": [], "eq_proposals_r": []}, {"eq_proposals_l": [], "eq_proposals_r": [], "subwoofer": [{"type": "peq"}]},
                {"eq_proposals_l": [], "eq_proposals_r": [], "subwoofer": [{"type": "peq", "fc_hz": "x", "gain_db": 0, "q": 1}]}):
        with pytest.raises(FilterExportError):
            compute_filters_digest(bad)


# --- JSON-Schema-Konformitaet (veroeffentlichtes Schema vs. Python-Validierung) -------
def _schema_validator():
    jsonschema = pytest.importorskip("jsonschema")  # optionale Dev-Abhaengigkeit; ueberspringt nur diese Tests
    schema = json.loads(SCHEMA.read_text("utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _valid_corpus():
    sub = [PeakingFilter("peq", 50.0, -3.0, 4.0), PeakingFilter("lowshelf", 60.0, 2.0, 0.7)]
    xo = {"hpf": {"frequency_hz": 25.0, "type": "24db-LR", "enabled": True}, "lpf": None, "phase_deg": 90}
    sat = [EqProposal(PeakingFilter("peq", 80.0, -2.0, 3.0), "mode")]
    return [
        _sub_doc(),
        _sub_doc(subwoofer_filters=sub, subwoofer_shelf_slopes_db={1: 12}, crossovers=xo, source_profile_sha256="b" * 64),
        _sub_doc(subwoofer_filters=[], intentional_flat=True),
        _sub_doc(no_harm=NoHarmResult(NoHarmStatus.UNSAFE, "x", 3)),
        _sub_doc(no_harm=NoHarmResult(NoHarmStatus.INCONCLUSIVE, "x", 0), checked_subwoofer_filters=None),
        _sub_doc(target_device="mosconi_pico_8_10", crossovers=xo),
        build_filter_export(run_id="r", target_device="kef_ls50_wireless", no_harm=SAFE, measurement_set_id="m", eq_proposals_l=sat, eq_proposals_r=sat),
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
    junk = [None, True, False, 0, 1, -1, 1.5, "", "x", [], {}, "a" * 64, "a" * 64 + "\n", 6, 12, 6.0, ["SAFE"]]

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
    lambda d: d["approval"].update(checked_filters_sha256="A" * 64),
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
