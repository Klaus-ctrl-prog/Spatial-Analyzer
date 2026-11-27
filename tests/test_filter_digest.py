"""Tests fuer filter_digest.py (synthetische Daten)."""

import copy
import hashlib
import math

import pytest

from spatial_analyzer.biquad_preview import PeakingFilter
from spatial_analyzer.filter_digest import (
    DIGEST_DOMAIN,
    FilterDigestError,
    canonical_float,
    filters_digest,
    peaking_filters_digest,
)


def _filters(l=None, r=None, sub=None):
    base = {"type": "peq", "fc_hz": 50.0, "gain_db": -3.0, "q": 4.0}
    return {
        "eq_proposals_l": [dict(base)] if l is None else l,
        "eq_proposals_r": [] if r is None else r,
        "subwoofer": [] if sub is None else sub,
    }


def test_negative_zero_and_zero_give_same_digest():
    """Regression G3: -0.0 und 0.0 duerfen bei fc, gain und q keinen unterschiedlichen Digest ergeben."""
    base = _filters()
    for field in ("fc_hz", "gain_db", "q"):
        a = copy.deepcopy(base)
        b = copy.deepcopy(base)
        a["eq_proposals_l"][0][field] = 0.0
        b["eq_proposals_l"][0][field] = -0.0
        assert filters_digest(a) == filters_digest(b), field
    assert canonical_float(-0.0) == 0.0 and math.copysign(1.0, canonical_float(-0.0)) == 1.0


def test_reason_is_ignored_by_digest():
    """Erlaeuterungstexte (reason) duerfen den Digest nicht veraendern, sonst bricht jede Umformulierung die Freigabe."""
    a = _filters()
    b = copy.deepcopy(a)
    b["eq_proposals_l"][0]["reason"] = "anderer Text"
    assert filters_digest(a) == filters_digest(b)


def test_shelf_slope_is_part_of_digest():
    """Shelf-Flanke 6, 12 und fehlend muessen unterschiedliche Digests ergeben (Flankenwahl veraendert das Ergebnis)."""
    def with_shelf(v):
        f = {"type": "lowshelf", "fc_hz": 60.0, "gain_db": 2.0, "q": 0.7}
        if v is not None:
            f["shelf_slope_db"] = v
        return _filters(l=[], sub=[f])

    digests = {filters_digest(with_shelf(v)) for v in (None, 6, 12)}
    assert len(digests) == 3


def test_order_and_list_membership_matter():
    """Reihenfolge innerhalb einer Liste und die Zugehoerigkeit zu einer Liste fliessen in den Digest ein."""
    f1 = {"type": "peq", "fc_hz": 50.0, "gain_db": -3.0, "q": 4.0}
    f2 = {"type": "peq", "fc_hz": 70.0, "gain_db": -2.0, "q": 3.0}
    assert filters_digest(_filters(l=[f1, f2])) != filters_digest(_filters(l=[f2, f1]))
    assert filters_digest(_filters(l=[f1])) != filters_digest(_filters(l=[], r=[f1]))
    assert filters_digest(_filters(l=[f1])) != filters_digest(_filters(l=[], sub=[f1]))


def test_int_and_float_values_give_same_digest():
    a = _filters(l=[{"type": "peq", "fc_hz": 50, "gain_db": -3, "q": 4}])
    b = _filters(l=[{"type": "peq", "fc_hz": 50.0, "gain_db": -3.0, "q": 4.0}])
    assert filters_digest(a) == filters_digest(b)


@pytest.mark.parametrize("bad", [
    float("nan"), float("inf"), float("-inf"), True, False, "50", 10 ** 400, -(10 ** 400), None,
])
def test_non_finite_bool_string_and_huge_values_rejected(bad):
    f = _filters(l=[{"type": "peq", "fc_hz": bad, "gain_db": -3.0, "q": 4.0}])
    with pytest.raises(FilterDigestError):
        filters_digest(f)
    with pytest.raises(FilterDigestError):
        canonical_float(bad)


def test_missing_required_key_and_unknown_key_rejected():
    for key in ("type", "fc_hz", "gain_db", "q"):
        row = {"type": "peq", "fc_hz": 50.0, "gain_db": -3.0, "q": 4.0}
        del row[key]
        with pytest.raises(FilterDigestError):
            filters_digest(_filters(l=[row]))
    row = {"type": "peq", "fc_hz": 50.0, "gain_db": -3.0, "q": 4.0, "extra": 1}
    with pytest.raises(FilterDigestError):
        filters_digest(_filters(l=[row]))


@pytest.mark.parametrize("shelf", [5, 0, 24, "6", True, False, 12.5, 8, "12", float("nan"), float("inf")])
def test_invalid_shelf_slope_rejected(shelf):
    """Regression: nur exakt 6/12 (int oder float) ist eine gueltige Flanke; alles andere wird abgewiesen."""
    row = {"type": "lowshelf", "fc_hz": 60.0, "gain_db": 2.0, "q": 0.7, "shelf_slope_db": shelf}
    with pytest.raises(FilterDigestError):
        filters_digest(_filters(sub=[row]))


def test_shelf_slope_12_and_12_0_give_same_digest():
    """Regression: 12 und 12.0 sind dieselbe Flanke und ergeben denselben Digest (kanonisch int)."""
    def with_shelf(v):
        return _filters(sub=[{"type": "lowshelf", "fc_hz": 60.0, "gain_db": 2.0, "q": 0.7, "shelf_slope_db": v}])

    assert filters_digest(with_shelf(12)) == filters_digest(with_shelf(12.0))
    assert filters_digest(with_shelf(6)) == filters_digest(with_shelf(6.0))
    assert filters_digest(with_shelf(6.0)) != filters_digest(with_shelf(12.0))


def test_shelf_slope_12_0_canonicalized_to_int_in_golden_text():
    """Golden: Flanke 12.0 erscheint im kanonischen Text als 12 (ohne Dezimalpunkt)."""
    import hashlib as _h

    text = (
        '{"domain":"spatial_analyzer.filters.v2","lists":{'
        '"eq_proposals_l":[],"eq_proposals_r":[],'
        '"subwoofer":[["lowshelf",60.0,2.0,0.7,12]]}}'
    )
    expected = _h.sha256(text.encode("utf-8")).hexdigest()
    filters = _filters(l=[], r=[], sub=[{"type": "lowshelf", "fc_hz": 60.0, "gain_db": 2.0, "q": 0.7, "shelf_slope_db": 12.0}])
    assert filters_digest(filters) == expected


def test_non_string_type_rejected_and_malformed_container_rejected():
    with pytest.raises(FilterDigestError):
        filters_digest(_filters(l=[{"type": ["peq"], "fc_hz": 50.0, "gain_db": -3.0, "q": 4.0}]))
    with pytest.raises(FilterDigestError):
        filters_digest([])
    with pytest.raises(FilterDigestError):
        filters_digest({"eq_proposals_l": [], "eq_proposals_r": []})  # subwoofer fehlt
    with pytest.raises(FilterDigestError):
        filters_digest({**_filters(), "eq_proposals_l": "x"})


def test_golden_vector_matches_independent_hand_built_canonical_string():
    """Erwartungswert wird HIER unabhaengig gebaut (kanonischer JSON-Text von Hand, dann hashlib),
    nicht durch filters_digest erzeugt."""
    canonical_text = (
        '{"domain":"spatial_analyzer.filters.v2","lists":{'
        '"eq_proposals_l":[["peq",50.0,-3.0,4.0,null]],'
        '"eq_proposals_r":[["lowshelf",60.0,2.5,0.7,12]],'
        '"subwoofer":[]}}'
    )
    expected = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    filters = _filters(
        l=[{"type": "peq", "fc_hz": 50.0, "gain_db": -3.0, "q": 4.0}],
        r=[{"type": "lowshelf", "fc_hz": 60.0, "gain_db": 2.5, "q": 0.7, "shelf_slope_db": 12}],
    )
    assert DIGEST_DOMAIN == "spatial_analyzer.filters.v2"
    assert filters_digest(filters) == expected


def test_peaking_filters_digest_matches_dict_form_and_normalizes_negative_zero():
    l = [PeakingFilter("peq", 50.0, -3.0, 4.0)]
    r = [PeakingFilter("peq", 70.0, -0.0, 3.0)]
    dict_form = _filters(
        l=[{"type": "peq", "fc_hz": 50.0, "gain_db": -3.0, "q": 4.0}],
        r=[{"type": "peq", "fc_hz": 70.0, "gain_db": 0.0, "q": 3.0}],
    )
    assert peaking_filters_digest(l, r) == filters_digest(dict_form)
