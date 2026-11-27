"""Tests fuer den Saxx-DS120-Adapter (synthetische Daten; keine echten Profile)."""

import copy
import dataclasses
import hashlib
import json

import pytest

from spatial_analyzer.adapters.saxx_ds120 import (
    AdapterBlockedError,
    MappingError,
    QuantizationError,
    ValidationError,
    assert_opaque_preserved,
    build_preview,
    compute_mapping_input_sha256,
    compute_mapping_output_sha256,
    load_lookup_tables,
    map_export_to_edit,
    verify_mapping,
    quantize,
    render_preview,
    validate_edit_json,
)
from spatial_analyzer.adapters.saxx_ds120.quantization import LOOKUP_FILE, LOOKUP_SHA256
from spatial_analyzer.adapters.saxx_ds120.validation import OPAQUE_BYTES
from spatial_analyzer.biquad_preview import PeakingFilter
from spatial_analyzer.eq_solver import NoHarmResult, NoHarmStatus
from spatial_analyzer.filter_export import build_filter_export

SAFE = NoHarmResult(NoHarmStatus.SAFE, "ok", 5)
T = load_lookup_tables()


def _doc(filters=None, no_harm=SAFE, crossovers=None, slopes=None):
    filters = [PeakingFilter("peq", 50.3, -3.1, 4.0)] if filters is None else filters
    return build_filter_export(
        run_id="r", target_device="saxx_ds120", no_harm=no_harm, measurement_set_id="m",
        subwoofer_filters=filters, crossovers=crossovers,
        subwoofer_shelf_slopes_db=slopes, intentional_flat=not filters and not crossovers)


# --- Lookup-Datei / Quantisierung -------------------------------------------------
def test_lookup_hash_and_counts():
    assert hashlib.sha256(LOOKUP_FILE.read_bytes()).hexdigest() == LOOKUP_SHA256
    assert LOOKUP_SHA256.upper() == "7A9B03FFE62D42068D612E1CC3FF078338FAD79317490B8EBF599F6B188D85E6"
    assert (len(T["frequency_hz"]), len(T["gain_db"]), len(T["q"])) == (244, 181, 79)


def test_modified_lookup_file_refused(tmp_path):
    doc = json.loads(LOOKUP_FILE.read_text("utf-8"))
    doc["tables"]["q"]["values"][0] += 0.001
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc), "utf-8")
    with pytest.raises(QuantizationError, match="SHA-256"):
        load_lookup_tables(p)


def test_quantize_nearest_documents_target_table_index_deviation():
    q = quantize(T["frequency_hz"], 20.5, "f")
    assert (q.target, q.table_value, q.index) == (20.5, 20.3, 1)
    assert q.deviation == pytest.approx(-0.2)
    exact = quantize(T["gain_db"], 0.0, "g")
    assert exact.table_value == 0.0 and exact.deviation == 0.0 and T["gain_db"][exact.index] == 0.0
    assert "->" in q.describe() and "Index 1" in q.describe()


@pytest.mark.parametrize("val", [19.0, 25000.0, float("nan"), float("inf"), True, "50"])
def test_quantize_rejects_out_of_range_and_non_numeric(val):
    with pytest.raises(QuantizationError):
        quantize(T["frequency_hz"], val, "f")


# --- Mapper -----------------------------------------------------------------------
def test_map_basic_peq_values_are_table_values_and_documented():
    res = build_preview(_doc())
    item = res.edit["eq"][0]
    assert item["band"] == 1 and item["type"] == "PEQ" and item["bypass"] is False
    assert item["frequency_hz"] in T["frequency_hz"] and item["gain_db"] in T["gain_db"] and item["q"] in T["q"]
    assert len(res.quantization) == 3 and res.released is False and res.hardware_validated is False
    validate_edit_json(res.edit, T)


def test_unused_bands_reset_1_9_bypass_10_14_gain_zero_and_optional():
    eq = {e["band"]: e for e in build_preview(_doc()).edit["eq"]}
    assert len(eq) == 14
    assert all(eq[b] == {"band": b, "bypass": True} for b in range(2, 10))
    assert all(eq[b] == {"band": b, "gain_db": 0.0} for b in range(10, 15))
    only = build_preview(_doc(), reset_unused_bands=False).edit["eq"]
    assert [e["band"] for e in only] == [1]


def test_shelves_go_to_bands_1_9_peqs_follow_and_10_14_only_peq():
    filters = [PeakingFilter("peq", 40.0, -2.0, 3.0), PeakingFilter("lowshelf", 60.0, 2.0, 0.7),
               PeakingFilter("highshelf", 8000.0, -1.0, 0.7)]
    res = build_preview(_doc(filters, slopes={1: 12, 2: 6}))
    eq = {e["band"]: e for e in res.edit["eq"]}
    assert (eq[1]["type"], eq[2]["type"], eq[3]["type"]) == ("LS12", "HS6", "PEQ")
    many = [PeakingFilter("peq", 30.0 + 5 * i, -1.0, 3.0) for i in range(14)]
    for e in build_preview(_doc(many)).edit["eq"]:
        if e["band"] >= 10:
            assert "type" not in e and "bypass" not in e


def test_shelf_without_slope_refused_and_too_many_filters_or_shelves():
    with pytest.raises(MappingError, match="shelf_slope_db"):
        build_preview(_doc([PeakingFilter("lowshelf", 60.0, 2.0, 0.7)]))
    with pytest.raises(MappingError, match="14"):
        build_preview(_doc([PeakingFilter("peq", 30.0 + 5 * i, -1.0, 3.0) for i in range(15)]))
    shelves = [PeakingFilter("lowshelf", 40.0 + 5 * i, 1.0, 0.7) for i in range(10)]
    with pytest.raises(MappingError, match="Shelf"):
        build_preview(_doc(shelves, slopes={i: 6 for i in range(10)}))


def test_out_of_range_values_refused_not_clamped():
    with pytest.raises(QuantizationError):
        build_preview(_doc([PeakingFilter("peq", 50.0, -30.0, 4.0)]))  # Gain < -24 dB
    with pytest.raises(QuantizationError):
        build_preview(_doc([PeakingFilter("peq", 50.0, -3.0, 200.0)]))  # Q ueber Tabellenende


def test_crossovers_phase_and_bypass_polarity():
    xo = {"hpf": {"frequency_hz": 25.0, "type": "24db-LR", "enabled": True},
          "lpf": {"frequency_hz": 80.0, "type": "24db-LR", "enabled": False}, "phase_deg": 135}
    res = build_preview(_doc(crossovers=xo))
    assert res.edit["phase_deg"] == 135
    assert res.edit["hpf"]["enabled"] is True and res.edit["lpf"]["enabled"] is False
    assert any("Bypass-Flag 0 (aktiv)" in n for n in res.notes) and any("Bypass-Flag 1 (Bypass)" in n for n in res.notes)
    for bad_phase in (91, 30, 360):
        with pytest.raises(MappingError, match="45"):
            build_preview(_doc(crossovers={**xo, "phase_deg": bad_phase}))
    with pytest.raises(MappingError, match="Weichentyp"):
        build_preview(_doc(crossovers={**xo, "hpf": {**xo["hpf"], "type": "36db-LR"}}))


def test_blocked_exports_not_mapped_but_previewable_and_marked():
    for status in (NoHarmStatus.UNSAFE, NoHarmStatus.INCONCLUSIVE):
        doc = _doc(no_harm=NoHarmResult(status, "x", 1))
        with pytest.raises(AdapterBlockedError):
            map_export_to_edit(doc)
        prev = build_preview(doc)
        assert prev.released is False
        text = render_preview(prev)
        assert "NICHT FREIGEGEBEN" in text and "hardware_validated=false" in text and "Index" in text


def test_build_preview_default_never_requires_release_and_marks_unreleased():
    doc = _doc(no_harm=NoHarmResult(NoHarmStatus.UNSAFE, "x", 1))
    prev = build_preview(doc)
    assert prev.released is False
    assert build_preview(_doc()).released is False  # Saxx: auch SAFE ist nie freigegeben


def test_build_preview_rejects_require_released_argument():
    # Regression G1: frueher TypeError "multiple values for keyword argument" aus map_export_to_edit;
    # jetzt bewusst ueber die explizite Preview-Signatur abgewiesen (True wie False).
    for value in (True, False):
        with pytest.raises(TypeError, match="unexpected keyword argument .require_released."):
            build_preview(_doc(), require_released=value)


def test_build_preview_regular_parameters_still_work():
    doc = _doc()
    res = build_preview(doc, profile_name="Vorschau", reset_unused_bands=False, tables=T)
    assert res.edit["name"] == "Vorschau"
    assert res.edit["eq"] == build_preview(doc, reset_unused_bands=False).edit["eq"]
    assert build_preview(doc, tables=T).edit == build_preview(doc).edit


def test_wrong_target_or_invalid_export_refused():
    doc = build_filter_export(run_id="r", target_device="mosconi_pico_8_10", no_harm=SAFE, measurement_set_id="m",
                              subwoofer_filters=[PeakingFilter("peq", 50.0, -3.0, 4.0)])
    with pytest.raises(MappingError, match="target_device"):
        build_preview(doc)
    bad = _doc()
    bad["approval"]["hardware_validated"] = True
    with pytest.raises(MappingError):
        build_preview(bad)


def test_satellite_filters_cannot_reach_saxx_edit():
    doc = _doc()
    doc["filters"]["eq_proposals_l"] = [{"type": "peq", "fc_hz": 80.0, "gain_db": -2.0, "q": 3.0}]
    with pytest.raises(MappingError, match="Satellitenfilter"):
        build_preview(doc)


def test_mapper_does_not_mutate_export_and_is_deterministic():
    doc = _doc()
    before = copy.deepcopy(doc)
    a, b = build_preview(doc), build_preview(doc)
    assert doc == before and a.edit == b.edit


def test_profile_name_validated():
    assert build_preview(_doc(), profile_name="SA test").edit["name"] == "SA test"
    with pytest.raises(ValidationError, match="name"):
        build_preview(_doc(), profile_name="x" * 20)
    with pytest.raises(ValidationError, match="name"):
        build_preview(_doc(), profile_name="Grüße")


# --- validate_edit_json / opaque --------------------------------------------------
@pytest.mark.parametrize("edit", [
    "x", {"unknown": 1}, {"phase_deg": 50}, {"eq": [{"band": 0}]}, {"eq": [{"band": 15}]},
    {"eq": [{"band": 1}, {"band": 1}]}, {"eq": [{"band": 10, "type": "LS6"}]},
    {"eq": [{"band": 10, "bypass": True}]}, {"eq": [{"band": 1, "type": "LS24"}]},
    {"eq": [{"band": 1, "gain_db": 0.1234}]}, {"hpf": {"enabled": 1}}, {"hpf": {"foo": 1}},
    {"eq": "x"}, {"eq": [{"band": 1, "opaque": 1}]},
])
def test_edit_validation_rejects(edit):
    with pytest.raises(ValidationError):
        validate_edit_json(edit, T)


def test_opaque_bytes_set_matches_spec_and_helper_detects_changes():
    assert OPAQUE_BYTES == {11, 16, 21, 25, *range(27, 34), *range(146, 170), 210, 211}
    tpl = bytearray(range(212))
    assert_opaque_preserved(tpl, bytearray(tpl))
    cand = bytearray(tpl)
    cand[12] ^= 0xFF
    cand[34] ^= 0xFF  # nicht-opaque Aenderungen sind erlaubt
    assert_opaque_preserved(tpl, cand)
    for idx in (11, 29, 146, 169, 211):
        cand = bytearray(tpl)
        cand[idx] ^= 1
        with pytest.raises(ValidationError, match=str(idx)):
            assert_opaque_preserved(tpl, cand)
    with pytest.raises(ValidationError):
        assert_opaque_preserved(tpl, bytearray(10))


def test_edit_json_never_addresses_opaque_regions():
    """Band 14 endet bei Byte 145; Bytes 146-169 sind nie als Extra-EQs adressierbar."""
    res = build_preview(_doc([PeakingFilter("peq", 30.0 + 5 * i, -1.0, 3.0) for i in range(14)]))
    assert max(e["band"] for e in res.edit["eq"]) == 14
    assert 34 + 13 * 8 + 8 - 1 < min(b for b in OPAQUE_BYTES if b >= 146)
    with pytest.raises(ValidationError):
        validate_edit_json({"eq": [{"band": 15}]}, T)


# --- Korrekturrunde: Randwerte, Gleichstaende, Bindung, Builder-Integration -----------
def _reference_nearest(table, t):
    """Referenzregel des Candidate Builders: naechster Wert, erster Index bei Gleichstand."""
    return min(range(len(table)), key=lambda i: abs(table[i] - t))


@pytest.mark.parametrize("name", ["frequency_hz", "gain_db", "q"])
def test_table_edges_and_every_table_value_are_exact(name):
    t = T[name]
    lo, hi = quantize(t, t[0], name), quantize(t, t[-1], name)
    assert (lo.index, lo.deviation, hi.index, hi.deviation) == (0, 0.0, len(t) - 1, 0.0)
    for i, v in enumerate(t):
        q = quantize(t, v, name)
        assert (q.index, q.table_value, q.deviation) == (i, v, 0.0)
    for outside in (t[0] - 1e-9, t[-1] + 1e-9, t[0] - 1, t[-1] + 1):
        with pytest.raises(QuantizationError):
            quantize(t, outside, name)


@pytest.mark.parametrize("name", ["frequency_hz", "gain_db", "q"])
def test_intermediate_values_and_midpoints_match_builder_rule(name):
    t = T[name]
    for i, (a, b) in enumerate(zip(t, t[1:])):
        mid = (a + b) / 2
        q = quantize(t, mid, name)
        assert q.index == _reference_nearest(t, mid) and q.index in (i, i + 1)
        assert abs(q.deviation) <= (b - a) / 2 + 1e-9
        for frac in (0.1, 0.4, 0.6, 0.9):  # klar naeher am unteren/oberen Nachbarn
            x = a + frac * (b - a)
            assert quantize(t, x, name).index == (i if frac < 0.5 else i + 1)


def test_exact_tie_chooses_first_index_on_synthetic_table():
    tbl = (1.0, 2.0, 3.0)
    assert quantize(tbl, 1.5, "t").index == 0 and quantize(tbl, 2.5, "t").index == 1
    assert quantize(tbl, 1.5, "t").deviation == -0.5 and quantize(tbl, 2.5, "t").deviation == -0.5


@pytest.mark.parametrize("val", [10 ** 400, -10 ** 400])
def test_quantize_huge_integers_raise_quantization_error(val):
    with pytest.raises(QuantizationError):
        quantize(T["gain_db"], val, "g")


def test_crossover_frequency_quantization_documented_and_range_checked():
    xo = {"hpf": {"frequency_hz": 25.2, "type": "24db-LR", "enabled": True},
          "lpf": {"frequency_hz": 80.0, "type": "12db-Bessel", "enabled": True}}
    res = build_preview(_doc(crossovers=xo))
    labels = {q.label: q for q in res.quantization}
    for key in ("HPF", "LPF"):
        q = labels[f"{key} Frequenz [Hz]"]
        assert res.edit[key.lower()]["frequency_hz"] == q.table_value in T["frequency_hz"]
        assert T["frequency_hz"][q.index] == q.table_value
    assert labels["HPF Frequenz [Hz]"].target == 25.2
    for bad in (19.0, 22000.0):
        with pytest.raises(QuantizationError):
            build_preview(_doc(crossovers={**xo, "lpf": {**xo["lpf"], "frequency_hz": bad}}))


def test_safe_export_with_uncovered_subwoofer_filters_is_blocked_by_adapter():
    f = [PeakingFilter("peq", 50.0, -3.0, 4.0)]
    doc = build_filter_export(run_id="r", target_device="saxx_ds120", no_harm=SAFE, measurement_set_id="m", subwoofer_filters=f)
    assert doc["approval"]["do_no_harm"] == "SAFE" and doc["approval"]["status"] == "BLOCKED"
    with pytest.raises(AdapterBlockedError, match="nicht an die exportierten Filter gebunden"):
        map_export_to_edit(doc)
    assert "nicht an die exportierten Filter gebunden" in render_preview(build_preview(doc))


def test_huge_phase_value_is_mapping_error_not_crash():
    for phase in (10 ** 400, -45, 360, 44):
        with pytest.raises(MappingError):
            build_preview(_doc(crossovers={"hpf": None, "lpf": None, "phase_deg": phase}))


def test_edit_validation_rejects_type_or_bypass_on_bands_10_to_14():
    for edit in ({"eq": [{"band": 10, "type": "PEQ"}]}, {"eq": [{"band": 14, "bypass": False}]}):
        with pytest.raises(ValidationError, match="weder Typ"):
            validate_edit_json(edit, T)


def test_tampered_released_export_is_refused_by_mapper():
    """Regression: ein per Hand auf RELEASED gesetzter Saxx-Export wird abgewiesen (keine Geraetefreigabe)."""
    doc = _doc()
    doc["approval"]["status"] = "RELEASED"
    with pytest.raises(MappingError, match="Saxx/Mosconi"):
        map_export_to_edit(doc)


# --- Integration mit dem ECHTEN Candidate Builder auf SYNTHETISCHEM Profil ------------
import importlib.util
import os
import sys
from pathlib import Path

_BUILDER_DEFAULT = (Path.home() / "Desktop" / "Saxx_DS120_Complete" / "Saxx_DS120_Validation_1_0"
                    / "saxx_datdsp_candidate_builder.py")
_BUILDER_PATH = Path(os.environ.get("SAXX_BUILDER_PATH", _BUILDER_DEFAULT))


def _load_builder():
    if not _BUILDER_PATH.is_file():
        pytest.skip(f"Candidate Builder nicht gefunden ({_BUILDER_PATH}); SAXX_BUILDER_PATH setzen.")
    spec = importlib.util.spec_from_file_location("saxx_builder_under_test", _BUILDER_PATH)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # keine __pycache__ im externen Projekt anlegen
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    if mod.__version__ != "0.15.0-qa":
        pytest.skip(f"Builder-Version {mod.__version__} != 0.15.0-qa")
    return mod


def _synthetic_template():
    """212 Byte mit gueltigem Header, unterscheidbaren opaque-Bytes - KEIN echtes Profil."""
    tpl = bytearray(212)
    tpl[0:4], tpl[6:8], tpl[9] = bytes.fromhex("A5A5A500"), b"\x00\x02", 0xC8
    for i in OPAQUE_BYTES:
        tpl[i] = (i * 7 + 3) & 0xFF
    for i in range(34, 146):
        tpl[i] = 0x55
    return tpl


def _u16(b, o):
    return b[o] | (b[o + 1] << 8)


def test_builder_quantizer_agrees_with_adapter_on_values_midpoints_and_random():
    B = _load_builder()
    import random
    rng = random.Random(5)
    for name, tbl in T.items():
        pts = list(tbl) + [(a + b) / 2 for a, b in zip(tbl, tbl[1:])] + [rng.uniform(tbl[0], tbl[-1]) for _ in range(500)]
        for x in pts:
            bi, bv = B._lookup_nearest(list(tbl), x, name)
            q = quantize(tbl, x, name)
            assert (q.index, q.table_value) == (bi, bv)
        for outside in (tbl[0] - 1e-6, tbl[-1] + 1e-6):
            with pytest.raises(B.ProfileError):
                B._lookup_nearest(list(tbl), outside, name)
            with pytest.raises(QuantizationError):
                quantize(tbl, outside, name)


def test_real_builder_accepts_mapper_output_on_synthetic_profile():
    B = _load_builder()
    filters = [PeakingFilter("peq", 33.3, -4.1, 6.2), PeakingFilter("lowshelf", 61.0, 2.35, 0.7),
               PeakingFilter("highshelf", 8200.0, -1.7, 0.9), PeakingFilter("peq", 47.0, -2.0, 9.9)]
    xo = {"hpf": {"frequency_hz": 22.0, "type": "24db-LR", "enabled": True},
          "lpf": {"frequency_hz": 115.0, "type": "24db-LR", "enabled": False}, "phase_deg": 225}
    res = build_preview(_doc(filters, slopes={1: 12, 2: 6}, crossovers=xo), profile_name="SA test")
    tpl = _synthetic_template()
    cand = bytearray(tpl)
    notes = B.apply_edits(cand, copy.deepcopy(res.edit), {k: list(v) for k, v in T.items()})
    assert len(notes) > 0
    assert_opaque_preserved(tpl, cand)  # opaque Bytes (inkl. 146-169) unveraendert
    assert bytes(cand[146:170]) == bytes(tpl[146:170])
    for e in res.edit["eq"]:  # vom Builder geschriebene Indizes == Adapter-Indizes
        off = 34 + (e["band"] - 1) * 8
        if "frequency_hz" in e:
            assert cand[off] == T["frequency_hz"].index(e["frequency_hz"])
        if "gain_db" in e:
            assert _u16(cand, off + 2) == T["gain_db"].index(e["gain_db"])
        if "q" in e:
            assert _u16(cand, off + 6) == T["q"].index(e["q"])
        if "bypass" in e:
            assert _u16(cand, off + 4) == (1 if e["bypass"] else 0)
    assert (cand[20], cand[24]) == (0, 1)  # HPF aktiv=0, LPF Bypass=1
    assert _u16(cand, 14) == (225 // 45) * 48
    assert T["frequency_hz"][cand[18]] == res.edit["hpf"]["frequency_hz"]
    assert T["frequency_hz"][cand[22]] == res.edit["lpf"]["frequency_hz"]
    # Bypass-Feld der Baender 10-14 schreibt der Builder nie
    assert bytes(cand[34 + 9 * 8 + 4:34 + 9 * 8 + 6]) == bytes(tpl[34 + 9 * 8 + 4:34 + 9 * 8 + 6])


def test_real_builder_rejects_what_adapter_validation_rejects():
    B = _load_builder()
    tbl = {k: list(v) for k, v in T.items()}
    bad_edits = [{"phase_deg": 50}, {"eq": [{"band": 10, "type": "PEQ"}]}, {"eq": [{"band": 15}]}, {"unknown": 1},
                 {"eq": [{"band": 1, "type": "LS24"}]}, {"hpf": {"type": "36db-LR"}}]
    for edit in bad_edits:
        with pytest.raises(B.ProfileError):
            B.apply_edits(_synthetic_template(), copy.deepcopy(edit), tbl)
        with pytest.raises(ValidationError):
            validate_edit_json(edit, T)


# --- Korrekturrunde B: Saxx nie freigegeben, reproduzierbare Mapping-Hashes ---------------
RELEASE_FILTERS = [PeakingFilter("peq", 50.3, -3.1, 4.0)]


def test_require_released_always_blocks_saxx_even_for_previously_covered_filters():
    """Regression: frueher RELEASED durch Selbstauskunft 'checked_subwoofer_filters'; jetzt immer AdapterBlockedError."""
    for filters in (RELEASE_FILTERS, [PeakingFilter("lowshelf", 60.0, 2.0, 0.7)]):
        doc = _doc(filters, slopes={0: 6} if filters[0].type == "lowshelf" else None)
        assert doc["approval"]["status"] == "BLOCKED"
        with pytest.raises(AdapterBlockedError, match="keine Geraetefreigabe ohne Subwoofer-Do-No-Harm-Modell"):
            map_export_to_edit(doc)
        with pytest.raises(AdapterBlockedError):
            map_export_to_edit(doc, require_released=True)


def test_mapping_hashes_are_reproducible_for_same_inputs():
    doc = _doc()
    a, b = build_preview(doc), build_preview(copy.deepcopy(doc))
    assert a.mapping_input_sha256 == b.mapping_input_sha256 and len(a.mapping_input_sha256) == 64
    assert a.mapping_output_sha256 == b.mapping_output_sha256 and len(a.mapping_output_sha256) == 64
    assert a.tables_sha256 == b.tables_sha256 and len(a.tables_sha256) == 64
    assert compute_mapping_output_sha256(a) == a.mapping_output_sha256


def test_mapping_input_hash_changes_with_every_relevant_input():
    """Regression: jede aenderbare Eingabe der Abbildung veraendert den Eingabe-Hash."""
    def h(doc=None, profile_name=None, reset=True, tables=None):
        d = _doc(slopes={1: 12, 2: 6}, crossovers={"hpf": {"frequency_hz": 25.0, "type": "24db-LR", "enabled": True},
                                                   "lpf": {"frequency_hz": 80.0, "type": "24db-LR", "enabled": False},
                                                   "phase_deg": 90}) if doc is None else doc
        return compute_mapping_input_sha256(d, profile_name=profile_name, reset_unused_bands=reset, tables=tables or T)

    base_filters = [PeakingFilter("peq", 40.0, -2.0, 3.0), PeakingFilter("lowshelf", 60.0, 2.0, 0.7),
                    PeakingFilter("highshelf", 8000.0, -1.0, 0.7)]
    base_xo = {"hpf": {"frequency_hz": 25.0, "type": "24db-LR", "enabled": True},
               "lpf": {"frequency_hz": 80.0, "type": "24db-LR", "enabled": False}, "phase_deg": 90}
    base = h(_doc(base_filters, slopes={1: 12, 2: 6}, crossovers=base_xo))
    variants = {
        "fc": [PeakingFilter("peq", 41.0, -2.0, 3.0), *base_filters[1:]],
        "gain": [PeakingFilter("peq", 40.0, -2.5, 3.0), *base_filters[1:]],
        "q": [PeakingFilter("peq", 40.0, -2.0, 3.5), *base_filters[1:]],
        "order": [base_filters[1], base_filters[0], base_filters[2]],
    }
    for name, filt in variants.items():
        slopes = {1: 12, 2: 6} if name != "order" else {0: 12, 2: 6}
        assert h(_doc(filt, slopes=slopes, crossovers=base_xo)) != base, name
    assert h(_doc([base_filters[0], PeakingFilter("lowshelf", 60.0, 2.0, 0.7), base_filters[2]],
                  slopes={1: 6, 2: 6}, crossovers=base_xo)) != base, "shelf slope"
    for name, xo in {
        "hpf freq": {**base_xo, "hpf": {**base_xo["hpf"], "frequency_hz": 30.0}},
        "hpf enabled": {**base_xo, "hpf": {**base_xo["hpf"], "enabled": False}},
        "phase": {**base_xo, "phase_deg": 135},
    }.items():
        assert h(_doc(base_filters, slopes={1: 12, 2: 6}, crossovers=xo)) != base, name
    assert h(_doc(base_filters, slopes={1: 12, 2: 6}, crossovers=base_xo), profile_name="X") != base, "profile"
    assert h(_doc(base_filters, slopes={1: 12, 2: 6}, crossovers=base_xo), reset=False) != base, "reset"
    custom = {k: list(v) for k, v in T.items()}
    custom["q"][0] = custom["q"][0] + 0.001
    assert h(_doc(base_filters, slopes={1: 12, 2: 6}, crossovers=base_xo), tables=custom) != base, "tables"


def test_negative_zero_in_mapped_value_gives_identical_hashes_g3():
    """Regression G3: -0.0 und 0.0 im Gain ergeben dieselben Eingabe- und Ausgabe-Hashes."""
    a = build_preview(_doc([PeakingFilter("peq", 50.3, 0.0, 4.0)]))
    b = build_preview(_doc([PeakingFilter("peq", 50.3, -0.0, 4.0)]))
    assert a.mapping_input_sha256 == b.mapping_input_sha256
    assert a.mapping_output_sha256 == b.mapping_output_sha256
    assert a.edit == b.edit


def test_verify_mapping_passes_for_untampered_result():
    doc = _doc()
    verify_mapping(doc, build_preview(doc))


def test_verify_mapping_detects_modified_edit():
    doc = _doc()
    res = build_preview(doc)
    res.edit["eq"][0]["gain_db"] = T["gain_db"][0]
    with pytest.raises(MappingError):
        verify_mapping(doc, res)


def test_verify_mapping_detects_replaced_output_hash():
    doc = _doc()
    res = build_preview(doc)
    res.mapping_output_sha256 = "0" * 64
    with pytest.raises(MappingError):
        verify_mapping(doc, res)


def test_verify_mapping_detects_changed_export():
    doc = _doc()
    res = build_preview(doc)
    doc["filters"]["subwoofer"][0]["gain_db"] = -9.0
    with pytest.raises(MappingError):
        verify_mapping(doc, res)


def test_hardware_validated_is_false_in_every_result():
    assert build_preview(_doc()).hardware_validated is False
    assert build_preview(_doc(no_harm=NoHarmResult(NoHarmStatus.UNSAFE, "x", 1))).hardware_validated is False


def test_shelf_slope_12_and_12_0_give_same_input_hash():
    """Regression N5: 12 und 12.0 (nach Validierung gleichwertig) ergeben denselben Eingabe-Hash."""
    d1 = _doc([PeakingFilter("lowshelf", 60.0, 2.0, 0.7)], slopes={0: 12})
    d2 = copy.deepcopy(d1)
    d2["filters"]["subwoofer"][0]["shelf_slope_db"] = 12.0
    h1 = compute_mapping_input_sha256(d1, profile_name=None, reset_unused_bands=True, tables=T)
    h2 = compute_mapping_input_sha256(d2, profile_name=None, reset_unused_bands=True, tables=T)
    assert h1 == h2


@pytest.mark.parametrize("slope", [12.5, True, "12", 8, float("nan")])
def test_input_hash_rejects_invalid_shelf_slope(slope):
    """Regression R1: compute_mapping_input_sha256 darf ungueltige Flanken nicht still per int() umwandeln."""
    d = _doc([PeakingFilter("lowshelf", 60.0, 2.0, 0.7)], slopes={0: 12})
    d["filters"]["subwoofer"][0]["shelf_slope_db"] = slope
    with pytest.raises(MappingError):
        compute_mapping_input_sha256(d, profile_name=None, reset_unused_bands=True, tables=T)


def test_input_hash_ignores_reason_run_id_measurement_set_and_do_no_harm():
    """Regression N7: reine Metadaten (Erlaeuterung, run_id, Messsatz, do_no_harm) veraendern den Eingabe-Hash nicht."""
    base = _doc()
    h0 = compute_mapping_input_sha256(base, profile_name=None, reset_unused_bands=True, tables=T)
    d = copy.deepcopy(base)
    d["run_id"] = "anderer-lauf"
    d["provenance"]["measurement_set_id"] = "anderer-messsatz"
    d["approval"]["do_no_harm"] = "UNSAFE"
    d["filters"]["subwoofer"][0]["reason"] = "anderer Text"
    assert compute_mapping_input_sha256(d, profile_name=None, reset_unused_bands=True, tables=T) == h0


def test_input_hash_changes_with_crossover_type():
    """Regression N7: Weichentyp ist Teil der Mapping-Eingabe."""
    xo = {"hpf": {"frequency_hz": 25.0, "type": "24db-LR", "enabled": True}, "lpf": None}
    other = {"hpf": {"frequency_hz": 25.0, "type": "12db-LR", "enabled": True}, "lpf": None}
    h_a = compute_mapping_input_sha256(_doc(crossovers=xo), profile_name=None, reset_unused_bands=True, tables=T)
    h_b = compute_mapping_input_sha256(_doc(crossovers=other), profile_name=None, reset_unused_bands=True, tables=T)
    assert h_a != h_b


def test_verify_mapping_detects_tampered_quantization_released_and_hardware_flag():
    """Regression N7: verify_mapping erkennt manipulierte Quantisierung, released=True und hardware_validated=True."""
    doc = _doc()
    res = build_preview(doc)
    q0 = res.quantization[0]
    tampered_q = dataclasses.replace(res, quantization=[dataclasses.replace(q0, index=q0.index + 1)] + res.quantization[1:])
    with pytest.raises(MappingError):
        verify_mapping(doc, tampered_q)
    with pytest.raises(MappingError):
        verify_mapping(doc, dataclasses.replace(res, released=True))
    with pytest.raises(MappingError):
        verify_mapping(doc, dataclasses.replace(res, hardware_validated=True))


def test_golden_mapping_input_sha256_hand_built_with_custom_tables():
    """Erwartungswert wird HIER von Hand als kanonischer Text gebaut und gehasht - nicht durch
    compute_mapping_input_sha256 erzeugt. Tabellen sind explizit klein gewaehlt."""
    tables = {"frequency_hz": (20.0, 50.0), "gain_db": (-3.0, 0.0), "q": (1.0, 4.0)}
    tables_text = '{"frequency_hz":[20.0,50.0],"gain_db":[-3.0,0.0],"q":[1.0,4.0]}'
    tables_hex = hashlib.sha256(tables_text.encode("utf-8")).hexdigest()
    text = (
        '{"builder_version":"0.15.0-qa","contract_version":"1.1",'
        '"crossovers":{"hpf":null,"lpf":null,"phase_deg":null},'
        '"domain":"spatial_analyzer.saxx_ds120.mapping_input.v1","mapper_version":"1",'
        '"profile_name":null,"reset_unused_bands":true,'
        '"subwoofer":[["peq",50.0,-3.0,4.0,null]],'
        '"tables_sha256":"' + tables_hex + '","target_device":"saxx_ds120"}'
    )
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    doc = {
        "contract_version": "1.1",
        "target_device": "saxx_ds120",
        "filters": {"subwoofer": [{"type": "peq", "fc_hz": 50.0, "gain_db": -3.0, "q": 4.0}]},
        "crossovers": {"hpf": None, "lpf": None},
    }
    assert compute_mapping_input_sha256(doc, profile_name=None, reset_unused_bands=True, tables=tables) == expected
