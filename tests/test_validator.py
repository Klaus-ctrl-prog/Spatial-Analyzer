import json

import numpy as np

from spatial_analyzer.validator import has_errors, validate_file, validate_input


def _good_entry(n=50):
    freq = np.linspace(20.0, 2000.0, n)
    spl = np.full_like(freq, 70.0)
    phase = np.zeros_like(freq)
    return {"freq_hz": freq.tolist(), "spl_db": spl.tolist(), "phase_deg": phase.tolist()}


def _good_positions(n_wander=3):
    positions = [{"id": "sofa_mitte", "l": _good_entry(), "r": _good_entry()}]
    positions += [{"id": f"wander_{i}", "l": _good_entry(), "r": _good_entry()} for i in range(n_wander)]
    return positions


class TestValidateInputValidData:
    def test_valid_input_has_no_errors(self):
        data = {"room_dimensions_m": [5.72, 3.72, 2.40], "positions": _good_positions(n_wander=8)}
        issues = validate_input(data)
        assert not has_errors(issues)

    def test_valid_input_with_8_wander_points_has_no_warnings(self):
        data = {"room_dimensions_m": [5.72, 3.72, 2.40], "positions": _good_positions(n_wander=8)}
        issues = validate_input(data)
        assert issues == []


class TestValidateInputStructuralErrors:
    def test_missing_positions_key_is_error(self):
        issues = validate_input({"room_dimensions_m": [1, 1, 1]})
        assert has_errors(issues)
        assert any("positions" in i.path for i in issues)

    def test_empty_positions_list_is_error(self):
        issues = validate_input({"positions": []})
        assert has_errors(issues)

    def test_root_not_a_dict_is_error(self):
        issues = validate_input(["not", "a", "dict"])
        assert has_errors(issues)

    def test_position_without_id_is_error(self):
        positions = _good_positions()
        del positions[0]["id"]
        issues = validate_input({"positions": positions})
        assert any(i.severity == "error" and "id" in i.path for i in issues)

    def test_position_without_l_or_r_is_error(self):
        positions = [{"id": "sofa_mitte"}]
        issues = validate_input({"positions": positions})
        assert has_errors(issues)

    def test_no_position_has_l_channel_is_error(self):
        positions = [{"id": "sofa_mitte", "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert any("Kanal l" in i.message for i in issues if i.severity == "error")

    def test_mismatched_freq_spl_length_is_error(self):
        entry = _good_entry()
        entry["spl_db"] = entry["spl_db"][:-5]
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert has_errors(issues)

    def test_mismatched_phase_length_is_error(self):
        entry = _good_entry()
        entry["phase_deg"] = entry["phase_deg"][:-5]
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert has_errors(issues)

    def test_invalid_room_dimensions_is_error(self):
        data = {"room_dimensions_m": [5.72, 3.72], "positions": _good_positions()}
        issues = validate_input(data)
        assert has_errors(issues)

    def test_negative_room_dimension_is_error(self):
        data = {"room_dimensions_m": [5.72, -3.72, 2.40], "positions": _good_positions()}
        issues = validate_input(data)
        assert has_errors(issues)

    def test_missing_impulse_response_sample_interval_is_error(self):
        entry = _good_entry()
        entry["impulse_response"] = {"samples": [0.1, 0.2, 0.3]}
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert has_errors(issues)

    def test_channel_not_a_dict_is_error(self):
        positions = [{"id": "sofa_mitte", "l": "nicht_ein_objekt", "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert has_errors(issues)

    def test_position_not_a_dict_is_error(self):
        issues = validate_input({"positions": ["nicht_ein_objekt"]})
        assert has_errors(issues)

    def test_freq_not_a_list_is_error(self):
        entry = _good_entry()
        entry["freq_hz"] = "20,40,60"
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert has_errors(issues)

    def test_non_numeric_freq_values_is_error(self):
        entry = _good_entry()
        entry["freq_hz"][0] = "zwanzig"
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert has_errors(issues)

    def test_phase_not_a_list_is_error(self):
        entry = _good_entry()
        entry["phase_deg"] = "0,0,0"
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert has_errors(issues)

    def test_impulse_response_not_a_dict_is_error(self):
        entry = _good_entry()
        entry["impulse_response"] = "nicht_ein_objekt"
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert has_errors(issues)

    def test_implausible_sample_interval_is_warning(self):
        entry = _good_entry()
        entry["impulse_response"] = {"samples": [0.1, 0.2, 0.3], "sample_interval_s": 60.0}
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert not has_errors(issues)
        assert any("sample_interval_s" in i.path for i in issues)


class TestValidateInputFuzzAndEdgeCases:
    """Haertung (Prioritaet 5): Fuzz-/Grenzwerttests fuer validator.py.
    NaN/Inf-Werte werden bewusst als `error` (nicht nur als vage
    "ausserhalb des plausiblen Bereichs"-`warning`) gemeldet, weil
    FrequencyGrid/interpolate_to_grid/ImpulseResponseData sie nach der
    gleichen Haertungsrunde explizit ablehnen (siehe grid.py/models.py) -
    run_pipeline() wuerde sonst erst dort abbrechen."""

    def test_nan_freq_value_is_error_not_just_range_warning(self):
        entry = _good_entry()
        entry["freq_hz"][0] = float("nan")
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert any(
            i.severity == "error" and "freq_hz" in i.path and "NaN" in i.message
            for i in issues
        )

    def test_inf_freq_value_is_error(self):
        entry = _good_entry()
        entry["freq_hz"][0] = float("inf")
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert has_errors(issues)

    def test_nan_spl_value_is_error(self):
        entry = _good_entry()
        entry["spl_db"][0] = float("nan")
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert any(
            i.severity == "error" and "spl_db" in i.path and "NaN" in i.message
            for i in issues
        )

    def test_nan_freq_does_not_also_trigger_monotonic_warning(self):
        # Ein NaN-Wert soll als klarer Fehler gemeldet werden, nicht
        # zusaetzlich (verwirrend) als "nicht streng monoton steigend" -
        # beide Pruefungen beruhen auf Vergleichen, die bei NaN immer False
        # sind, sollen sich hier aber nicht ueberlagern.
        entry = _good_entry()
        entry["freq_hz"][5] = float("nan")
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert not any("monoton" in i.message for i in issues)

    def test_nan_sample_interval_is_error(self):
        entry = _good_entry()
        entry["impulse_response"] = {"samples": [0.1, 0.2, 0.3], "sample_interval_s": float("nan")}
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert any(
            i.severity == "error" and "sample_interval_s" in i.path and "NaN" in i.message
            for i in issues
        )

    def test_inf_sample_interval_is_error(self):
        entry = _good_entry()
        entry["impulse_response"] = {"samples": [0.1, 0.2, 0.3], "sample_interval_s": float("inf")}
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert has_errors(issues)

    def test_non_numeric_impulse_response_samples_is_error(self):
        # Vorher unbemerkt: ImpulseResponseData/np.asarray(dtype=float64)
        # wuerde erst in der Pipeline selbst mit einem rohen ValueError
        # abbrechen.
        entry = _good_entry()
        entry["impulse_response"] = {"samples": ["a", "b", "c"], "sample_interval_s": 1 / 48000}
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert any(
            i.severity == "error" and "impulse_response.samples" in i.path for i in issues
        )

    def test_boolean_sample_interval_is_error(self):
        # isinstance(True, int) ist True in Python - ohne expliziten
        # bool-Ausschluss wuerde ein boolescher Wert faelschlich als
        # gueltige Zahl durchgehen.
        entry = _good_entry()
        entry["impulse_response"] = {"samples": [0.1, 0.2, 0.3], "sample_interval_s": True}
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert has_errors(issues)

    def test_duplicate_freq_values_trigger_monotonic_warning(self):
        entry = _good_entry()
        freq = entry["freq_hz"]
        freq[11] = freq[10]  # exaktes Duplikat statt Vertauschung
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert any("monoton" in i.message for i in issues)

    def test_extreme_spl_outlier_is_warning_not_crash(self):
        entry = _good_entry()
        entry["spl_db"][0] = 1e300
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert not has_errors(issues)
        assert any("spl_db" in i.path for i in issues)

    def test_positions_as_dict_instead_of_list_is_error_not_crash(self):
        issues = validate_input({"positions": {"sofa_mitte": {}}})
        assert has_errors(issues)

    def test_tiny_sliver_frequency_range_does_not_crash(self):
        entry = {"freq_hz": [100.0, 100.0001], "spl_db": [70.0, 70.0]}
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert isinstance(issues, list)


class TestValidateInputWarnings:
    def test_missing_sofa_mitte_is_warning_not_error(self):
        positions = [{"id": "wander_0", "l": _good_entry(), "r": _good_entry()}]
        positions += [{"id": "wander_1", "l": _good_entry(), "r": _good_entry()}]
        positions += [{"id": "wander_2", "l": _good_entry(), "r": _good_entry()}]
        positions += [{"id": "wander_3", "l": _good_entry(), "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert not has_errors(issues)
        assert any("sofa_mitte" in i.message for i in issues if i.severity == "warning")

    def test_too_few_positions_is_warning(self):
        issues = validate_input({"positions": _good_positions(n_wander=1)})
        assert any("Position(en)" in i.message for i in issues if i.severity == "warning")

    def test_duplicate_id_is_warning(self):
        positions = _good_positions(n_wander=2)
        positions[2]["id"] = positions[0]["id"]
        issues = validate_input({"positions": positions})
        assert any("doppelte id" in i.message for i in issues)

    def test_implausible_spl_value_is_warning(self):
        entry = _good_entry()
        entry["spl_db"][0] = 500.0  # offensichtlicher Tippfehler
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        positions += _good_positions(n_wander=3)
        issues = validate_input({"positions": positions})
        assert not has_errors(issues)
        assert any("plausiblen Bereich" in i.message and "spl_db" in i.path for i in issues)

    def test_non_monotonic_freq_is_warning(self):
        entry = _good_entry()
        freq = entry["freq_hz"]
        freq[10], freq[11] = freq[11], freq[10]
        positions = [{"id": "sofa_mitte", "l": entry, "r": _good_entry()}]
        issues = validate_input({"positions": positions})
        assert any("monoton" in i.message for i in issues)


class TestValidateFile:
    def test_valid_file_has_no_errors(self, tmp_path):
        path = tmp_path / "input.json"
        data = {"room_dimensions_m": [5.72, 3.72, 2.40], "positions": _good_positions(n_wander=8)}
        path.write_text(json.dumps(data), encoding="utf-8")
        issues = validate_file(path)
        assert not has_errors(issues)

    def test_invalid_json_is_single_error(self, tmp_path):
        path = tmp_path / "input.json"
        path.write_text("{ nicht valides json", encoding="utf-8")
        issues = validate_file(path)
        assert has_errors(issues)
        assert len(issues) == 1

    def test_missing_file_is_single_error(self, tmp_path):
        issues = validate_file(tmp_path / "does_not_exist.json")
        assert has_errors(issues)
        assert len(issues) == 1


class TestValidationIssueStr:
    def test_str_marks_error_as_fehler(self):
        from spatial_analyzer.validator import ValidationIssue
        issue = ValidationIssue("error", "positions", "fehlt")
        assert str(issue) == "[FEHLER] positions: fehlt"

    def test_str_marks_warning_as_hinweis(self):
        from spatial_analyzer.validator import ValidationIssue
        issue = ValidationIssue("warning", "positions", "ungewoehnlich")
        assert str(issue) == "[Hinweis] positions: ungewoehnlich"
