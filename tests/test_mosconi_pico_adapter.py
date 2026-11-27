"""Tests fuer den Mosconi-Pico-Adapter (synthetische Daten)."""

import copy
import importlib
import sys

import pytest

from spatial_analyzer.adapters.mosconi_pico import MosconiValidationError, map_channels, validate_channel
from spatial_analyzer.adapters.mosconi_pico.capabilities import DELAY_GRID_MS
from spatial_analyzer.adapters.mosconi_pico.validation import delay_grid_value, delay_to_steps


def _tw(**kw):
    ch = {"name": "TW-L", "role": "tweeter", "delay_ms": 0.5, "phase_deg": 0,
          "hpf": {"frequency_hz": 2500.0, "slope": "24dB-LR"}, "peq": []}
    ch.update(kw)
    return ch


def test_grid_is_002_not_001():
    assert DELAY_GRID_MS == 0.02


@pytest.mark.parametrize("want,steps,grid", [(0.0, 0, 0.0), (0.5, 25, 0.5), (0.51, 26, 0.52), (0.49, 25, 0.5), (1.0, 50, 1.0)])
def test_delay_quantized_to_002_grid_with_deviation(want, steps, grid):
    row = map_channels([_tw(delay_ms=want)])[0]
    assert row["delay_steps"] == steps and row["delay_ms"] == grid
    assert row["delay_deviation_ms"] == pytest.approx(grid - want, abs=1e-9)
    assert delay_to_steps(want) == steps


def test_valid_mapping_does_not_mutate_input():
    chans = [_tw(), {"name": "SUB", "role": "subwoofer", "delay_ms": 1.0, "phase_deg": 180,
                     "lpf": {"frequency_hz": 80.0, "slope": "12dB-LR"},
                     "peq": [{"fc_hz": 50.0, "gain_db": -3.0, "q": 4.0}]}]
    before = copy.deepcopy(chans)
    rows = map_channels(chans)
    assert chans == before and [r["name"] for r in rows] == ["TW-L", "SUB"]
    assert rows[1]["phase_deg"] == 180 and rows[1]["lpf"]["slope"] == "12dB-LR"


@pytest.mark.parametrize("hpf", [None, {"frequency_hz": 2499.9, "slope": "12dB-LR"}, {"frequency_hz": 1000.0, "slope": "12dB-LR"}])
def test_tweeter_hpf_protection(hpf):
    with pytest.raises(MosconiValidationError, match="Hochtoener"):
        map_channels([_tw(hpf=hpf)])


def test_tweeter_hpf_exactly_2500_ok():
    assert map_channels([_tw(hpf={"frequency_hz": 2500.0, "slope": "12dB-LR"})])


def test_peq_limit_six_per_channel():
    peq = [{"fc_hz": 50.0 + i, "gain_db": -1.0, "q": 2.0} for i in range(6)]
    assert map_channels([_tw(peq=peq)])
    with pytest.raises(MosconiValidationError, match="maximal 6"):
        map_channels([_tw(peq=peq + [{"fc_hz": 90.0, "gain_db": -1.0, "q": 2.0}])])


@pytest.mark.parametrize("field,value", [
    ("phase_deg", 90), ("phase_deg", 45), ("phase_deg", True), ("delay_ms", -0.02), ("delay_ms", float("nan")),
    ("role", "tweeter2"), ("bogus", 1), ("name", ""),
    ("lpf", {"frequency_hz": 80.0, "slope": "18dB-LR"}), ("lpf", {"frequency_hz": 80.0, "slope": "24dB-Bworth"}),
    ("lpf", {"frequency_hz": -1.0, "slope": "12dB-LR"}), ("lpf", {"slope": "12dB-LR"}),
    ("peq", [{"fc_hz": 50.0, "gain_db": -1.0}]), ("peq", [{"fc_hz": 50.0, "gain_db": -1.0, "q": 0}]),
    ("peq", "x"),
])
def test_invalid_channel_fields_rejected(field, value):
    assert validate_channel(_tw(**{field: value}))
    with pytest.raises(MosconiValidationError):
        map_channels([_tw(**{field: value})])


def test_empty_non_list_and_duplicate_names_rejected():
    for bad in ([], None, "x", [1]):
        with pytest.raises(MosconiValidationError):
            map_channels(bad)
    with pytest.raises(MosconiValidationError, match="mehrfach"):
        map_channels([_tw(), _tw()])


def test_no_dependency_on_smart453_namespace_or_core_calculation():
    for name in [m for m in sys.modules if m.startswith("spatial_analyzer.adapters.mosconi_pico")]:
        del sys.modules[name]
    importlib.import_module("spatial_analyzer.adapters.mosconi_pico")
    assert not any(m.startswith("smart453_bass") for m in sys.modules)
    mod = sys.modules["spatial_analyzer.adapters.mosconi_pico.mapper"]
    assert not hasattr(mod, "propose_mode_cuts")


# --- Korrekturrunde: Gleichstaende, Overflow, Typen -----------------------------------
@pytest.mark.parametrize("want,steps,grid", [
    (0.01, 1, 0.02), (0.03, 2, 0.04), (0.05, 3, 0.06), (0.07, 4, 0.08), (0.09, 5, 0.1),  # Gleichstand -> aufrunden
    (0.0, 0, 0.0), (0.0099, 0, 0.0), (0.0101, 1, 0.02), (0.02, 1, 0.02), (0.1, 5, 0.1), (12.34, 617, 12.34),
])
def test_tie_rule_is_half_up_on_decimal_value(want, steps, grid):
    row = map_channels([_tw(delay_ms=want)])[0]
    assert row["delay_steps"] == delay_to_steps(want) == steps
    assert row["delay_ms"] == grid


def test_deviation_is_exact_grid_minus_target():
    assert map_channels([_tw(delay_ms=0.51)])[0]["delay_deviation_ms"] == 0.01
    assert map_channels([_tw(delay_ms=0.03)])[0]["delay_deviation_ms"] == 0.01
    assert map_channels([_tw(delay_ms=0.5)])[0]["delay_deviation_ms"] == 0.0


def test_huge_delay_does_not_overflow_and_is_exact():
    row = map_channels([_tw(delay_ms=1e308)])[0]  # kein OverflowError; Geraetemaximum ist nicht belegt
    assert isinstance(row["delay_steps"], int) and row["delay_steps"] > 10 ** 300


@pytest.mark.parametrize("val", [10 ** 400, -10 ** 400, float("inf"), float("-inf"), float("nan"), True, "0.5", None])
def test_non_finite_or_non_numeric_delay_is_validation_error(val):
    with pytest.raises(MosconiValidationError):
        map_channels([_tw(delay_ms=val)])


_BAD_DELAYS = [10 ** 400, -10 ** 400, 10 ** 309, float("inf"), float("-inf"), float("nan"), True, False, "0.5", None]


@pytest.mark.parametrize("val", _BAD_DELAYS)
def test_delay_helpers_reject_invalid_input_with_validation_error(val):
    # Regression G2: direkter Aufruf der Hilfsfunktionen warf OverflowError/TypeError/ValueError uneinheitlich.
    with pytest.raises(MosconiValidationError):
        delay_to_steps(val)
    with pytest.raises(MosconiValidationError):
        delay_grid_value(val)


@pytest.mark.parametrize("want,steps", [(0.0, 0), (0, 0), (0.01, 1), (0.03, 2), (0.05, 3), (0.5, 25), (1, 50), (1e308, None)])
def test_delay_helpers_unchanged_for_valid_input(want, steps):
    got_steps, grid, deviation = delay_grid_value(want)
    assert delay_to_steps(want) == got_steps
    if steps is not None:
        assert got_steps == steps
    else:
        assert isinstance(got_steps, int) and got_steps > 10 ** 300  # wie bisher: kein Geraetemaximum belegt


@pytest.mark.parametrize("name", [[1], {"a": 1}, 5, None, 1.5, ("a",)])
def test_unhashable_or_non_string_names_are_validation_errors(name):
    with pytest.raises(MosconiValidationError):
        map_channels([_tw(name=name), _tw(name=name)])
    with pytest.raises(MosconiValidationError):
        map_channels([_tw(name=name)])


@pytest.mark.parametrize("phase", [0.0, 180.0, "0", None, 180.5])
def test_phase_must_be_integer_0_or_180(phase):
    with pytest.raises(MosconiValidationError):
        map_channels([_tw(phase_deg=phase)])
    assert map_channels([_tw(phase_deg=0)]) and map_channels([_tw(phase_deg=180)])
