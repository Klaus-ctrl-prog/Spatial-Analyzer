"""Regressionstests zu den Befunden des unabhaengigen Reviews (Schritt 3, 28.11.2026).

F1: gelockerte Do-No-Harm-Toleranzen duerfen keine KEF-Freigabe begruenden.
F3: crossovers sind nicht im Pruefdigest -> KEF-RELEASED nur ohne crossovers.
F4: Subwoofer-Exporte tragen nie das KEF-Urteil SAFE.
Rein synthetische Daten.
"""

import copy
import json

import numpy as np
import pytest

from spatial_analyzer.biquad_preview import PeakingFilter
from spatial_analyzer.eq_solver import EqProposal, validate_no_harm
from spatial_analyzer.filter_export import (
    FilterExportError,
    build_filter_export,
    is_device_released,
    validate_filter_export,
)
from spatial_analyzer.pipeline import run_pipeline
from tests.test_pipeline import _synthetic_position_with_sub

FILTERS = [PeakingFilter("peq", 100.0, -4.0, 1.0)]


def _no_harm(**kw):
    freq = np.linspace(90.0, 110.0, 21)
    phase = np.deg2rad(180.0 + np.array([0.0, -5.0, 5.0]))[:, None]
    l_c = np.ones((3, freq.size), dtype=np.complex128)
    r_c = 0.5 * np.exp(1j * phase) * np.ones((3, freq.size))
    return validate_no_harm(freq, ["p1", "p2", "p3"], l_c, r_c, FILTERS, FILTERS, **kw)


def _kef(no_harm, **kw):
    sat = [EqProposal(f, "mode") for f in FILTERS]
    return build_filter_export(run_id="r", target_device="kef_ls50_wireless", no_harm=no_harm,
                               measurement_set_id="m", eq_proposals_l=sat, eq_proposals_r=sat, **kw)


def test_default_check_is_released():
    nh = _no_harm()
    assert nh.checked_default_tolerances is True
    assert _kef(nh)["approval"]["status"] == "RELEASED"


@pytest.mark.parametrize("kw", [
    {"tolerance_db": 1e9},
    {"local_tolerance_db": 1e9},
    {"target_tolerance_db": 1e9},
    {"tolerance_db": 0.4},  # auch strenger ist "nicht Standard" -> keine Freigabe
    {"min_positions": 1},
])
def test_non_default_tolerances_never_release(kw):
    nh = _no_harm(**kw)
    assert nh.checked_default_tolerances is False
    doc = _kef(nh)
    assert doc["approval"]["status"] == "BLOCKED"
    assert not is_device_released(doc)


def test_crossovers_block_kef_release_in_builder():
    for xo in ({"hpf": None, "lpf": None, "phase_deg": 180},
               {"hpf": {"type": "LR24", "frequency_hz": 80.0}, "lpf": None}):
        try:
            doc = _kef(_no_harm(), crossovers=xo)
        except FilterExportError:
            continue  # ungueltige Weiche wird ohnehin abgewiesen
        assert doc["approval"]["status"] == "BLOCKED"


def test_crossovers_added_after_release_invalidate_document():
    doc = _kef(_no_harm())
    assert is_device_released(doc)
    tampered = copy.deepcopy(doc)
    tampered["crossovers"]["phase_deg"] = 180
    with pytest.raises(FilterExportError, match="crossovers"):
        validate_filter_export(tampered)
    with pytest.raises(FilterExportError):  # fail-closed: kein stilles True
        is_device_released(tampered)


def test_pipeline_saxx_export_never_carries_kef_safe(tmp_path):
    positions = [_synthetic_position_with_sub(f"p{i}") for i in range(4)]
    (tmp_path / "in.json").write_text(json.dumps({"positions": positions}), encoding="utf-8")
    run_pipeline(tmp_path / "in.json", tmp_path / "out")
    saxx = json.loads((tmp_path / "out" / "filter_export_saxx.json").read_text("utf-8"))
    assert saxx["approval"]["do_no_harm"] == "INCONCLUSIVE"
    assert saxx["approval"]["status"] == "BLOCKED"
