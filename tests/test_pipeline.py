import json

import numpy as np
import pytest

from spatial_analyzer.pipeline import (
    _build_id_index_map,
    _build_joint_lr_data,
    _interpolate_channel_to_common_grid,
    _unwrap_phase_deg,
    run_pipeline,
)


def _synthetic_ir_with_slow_decay(fs=4000.0, n=8000, f0=60.0, true_tau_60db=1.0):
    dt = 1.0 / fs
    t = np.arange(n) * dt
    slope_db_per_s = -60.0 / true_tau_60db
    amplitude = 10 ** ((slope_db_per_s * t) / 20.0)
    samples = amplitude * np.sin(2 * np.pi * f0 * t)
    return {"samples": samples.tolist(), "sample_interval_s": dt}


def _synthetic_position(position_id, with_ir=False):
    freq = np.linspace(20.0, 2000.0, 400)
    spl = np.full_like(freq, 70.0)
    # Konsistenter Peak bei ~60 Hz (Lorentz-Form), an jeder Position vorhanden.
    spl += 8.0 / (1.0 + ((freq - 60.0) / 3.0) ** 2)
    phase = np.zeros_like(freq)

    l_entry = {"freq_hz": freq.tolist(), "spl_db": spl.tolist(), "phase_deg": phase.tolist()}
    if with_ir:
        l_entry["impulse_response"] = _synthetic_ir_with_slow_decay()

    r_entry = {"freq_hz": freq.tolist(), "spl_db": spl.tolist(), "phase_deg": phase.tolist()}
    return {"id": position_id, "l": l_entry, "r": r_entry}


def _write_synthetic_input(path):
    positions = [_synthetic_position("sofa_mitte", with_ir=True)]
    positions += [_synthetic_position(f"wander_{i}") for i in range(3)]
    data = {"room_dimensions_m": [5.72, 3.72, 2.40], "positions": positions}
    path.write_text(json.dumps(data), encoding="utf-8")


def _synthetic_position_with_sub(position_id):
    pos = _synthetic_position(position_id)
    sub_freq = np.linspace(20.0, 2000.0, 400)
    sub_spl = np.full_like(sub_freq, 65.0)
    sub_phase = np.zeros_like(sub_freq)
    pos["sub"] = {
        "freq_hz": sub_freq.tolist(), "spl_db": sub_spl.tolist(), "phase_deg": sub_phase.tolist()
    }
    return pos


class TestRunPipeline:
    def test_produces_expected_output_files(self, tmp_path):
        input_path = tmp_path / "input.json"
        _write_synthetic_input(input_path)
        output_dir = tmp_path / "out"

        report = run_pipeline(input_path, output_dir)

        for name in (
            "report.json", "report.html", "frequency_table_l.csv", "frequency_table_r.csv",
            "kefplayer_profile.json", "rew_eq_L.txt", "rew_eq_R.txt",
        ):
            assert (output_dir / name).exists(), f"{name} fehlt"

        assert report.room_dimensions_m == (5.72, 3.72, 2.40)

    def test_consistent_confirmed_mode_yields_cut_proposal(self, tmp_path):
        input_path = tmp_path / "input.json"
        _write_synthetic_input(input_path)
        report = run_pipeline(input_path, tmp_path / "out")

        assert len(report.mode_confirmations) >= 1
        mode_60 = min(report.mode_confirmations, key=lambda c: abs(c.freq_hz - 60.0))
        assert mode_60.classification in ("mode_candidate", "mode_candidate_unmodeled")
        assert mode_60.decay_confirmed is True

        cut_proposals = [p for p in report.eq_proposals_l if abs(p.filter.fc - 60.0) < 5.0]
        assert len(cut_proposals) >= 1
        assert all(p.filter.gain_db < 0 for p in cut_proposals)

    def test_missing_reference_ir_adds_note_and_skips_decay(self, tmp_path):
        positions = [_synthetic_position("sofa_mitte", with_ir=False)]
        positions += [_synthetic_position(f"wander_{i}") for i in range(2)]
        data = {"room_dimensions_m": [5.72, 3.72, 2.40], "positions": positions}
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(data), encoding="utf-8")

        report = run_pipeline(input_path, tmp_path / "out")
        assert any("Impulsantwort" in n for n in report.notes)
        assert all(c.decay_confirmed is None for c in report.mode_confirmations)
        # Ohne Abkling-Bestaetigung duerfen keine Mode-Cuts entstehen.
        mode_cuts = [p for p in report.eq_proposals_l if "Mode" in p.reason or "mode" in p.reason.lower()]
        assert mode_cuts == []

    def test_raises_on_empty_positions(self, tmp_path):
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps({"positions": []}), encoding="utf-8")
        with pytest.raises(ValueError):
            run_pipeline(input_path, tmp_path / "out")

    def test_do_no_harm_note_present_when_phase_available(self, tmp_path):
        input_path = tmp_path / "input.json"
        _write_synthetic_input(input_path)
        report = run_pipeline(input_path, tmp_path / "out")
        assert any("Do-No-Harm" in n for n in report.notes)

    def test_sub_measurements_produce_optimizer_result(self, tmp_path):
        positions = [_synthetic_position_with_sub("sofa_mitte")]
        positions += [_synthetic_position_with_sub(f"wander_{i}") for i in range(2)]
        data = {"room_dimensions_m": [5.72, 3.72, 2.40], "positions": positions}
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(data), encoding="utf-8")

        report = run_pipeline(input_path, tmp_path / "out")

        assert report.sub_optimizer_result is not None
        assert report.sub_optimizer_result.mode_band_hz == (60.0, 62.0)
        assert not any("Keine vollstaendigen Sub-Messungen" in n for n in report.notes)

    def test_missing_sub_on_one_position_notes_and_skips(self, tmp_path):
        positions = [_synthetic_position_with_sub("sofa_mitte")]
        positions.append(_synthetic_position("wander_0"))  # kein "sub"
        data = {"room_dimensions_m": [5.72, 3.72, 2.40], "positions": positions}
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(data), encoding="utf-8")

        report = run_pipeline(input_path, tmp_path / "out")

        assert report.sub_optimizer_result is None
        assert any("Keine vollstaendigen Sub-Messungen" in n for n in report.notes)

    def test_reads_notes_per_position_and_channel(self, tmp_path):
        positions = [_synthetic_position("sofa_mitte", with_ir=True)]
        positions[0]["notes"] = "KEF Highpass 90 Hz, Phasenkorrektur ein"
        positions[0]["l"]["notes"] = "Saxx Phase 180 Grad, Subpegel -5 dB"
        positions += [_synthetic_position(f"wander_{i}") for i in range(2)]
        data = {"room_dimensions_m": [5.72, 3.72, 2.40], "positions": positions}
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(data), encoding="utf-8")

        report = run_pipeline(input_path, tmp_path / "out")

        position_level = [n for n in report.measurement_notes if n.channel is None]
        channel_level = [n for n in report.measurement_notes if n.channel == "l"]
        assert position_level and position_level[0].text == "KEF Highpass 90 Hz, Phasenkorrektur ein"
        assert position_level[0].position_id == "sofa_mitte"
        assert channel_level and channel_level[0].text == "Saxx Phase 180 Grad, Subpegel -5 dB"
        # Streng getrennt von den System-Hinweisen:
        assert "KEF Highpass 90 Hz, Phasenkorrektur ein" not in report.notes

    def test_missing_notes_field_is_backward_compatible(self, tmp_path):
        input_path = tmp_path / "input.json"
        _write_synthetic_input(input_path)  # enthaelt kein "notes"-Feld

        report = run_pipeline(input_path, tmp_path / "out")

        assert report.measurement_notes == []

    def test_blank_notes_field_is_ignored(self, tmp_path):
        positions = [_synthetic_position("sofa_mitte", with_ir=True)]
        positions[0]["notes"] = "   "
        positions += [_synthetic_position(f"wander_{i}") for i in range(2)]
        data = {"room_dimensions_m": [5.72, 3.72, 2.40], "positions": positions}
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(data), encoding="utf-8")

        report = run_pipeline(input_path, tmp_path / "out")

        assert report.measurement_notes == []


def _entry(freq, spl=None, phase=None):
    freq = list(freq)
    n = len(freq)
    return {
        "freq_hz": freq,
        "spl_db": spl if spl is not None else [70.0] * n,
        "phase_deg": phase if phase is not None else [0.0] * n,
    }


class TestBuildIdIndexMap:
    def test_unique_ids_map_to_correct_rows(self):
        assert _build_id_index_map(["a", "b", "c"]) == {"a": 0, "b": 1, "c": 2}

    def test_duplicate_ids_raise_value_error(self):
        with pytest.raises(ValueError, match="doppelte"):
            _build_id_index_map(["a", "b", "a"])

    def test_empty_list_yields_empty_map(self):
        assert _build_id_index_map([]) == {}


class TestUnwrapPhaseDeg:
    def test_wrapped_179_to_minus_179_becomes_continuous(self):
        # Ein echter Sprung von +179 Grad auf +181 Grad (gewrappt auf
        # -179 Grad abgelegt) muss nach dem Entpacken ~181 Grad sein, nicht
        # ~-179 Grad.
        result = _unwrap_phase_deg([179.0, -179.0])
        assert result[1] == pytest.approx(181.0)

    def test_no_jump_stays_unchanged(self):
        result = _unwrap_phase_deg([10.0, 20.0, 30.0])
        np.testing.assert_allclose(result, [10.0, 20.0, 30.0])


class TestInterpolateChannelToCommonGridPhaseUnwrap:
    def test_phase_wrap_does_not_interpolate_through_zero(self):
        # Die Phase dreht in Wirklichkeit von +179 auf +181 Grad weiter,
        # abgelegt (gewrappt) als -179 Grad - eine naive lineare
        # Interpolation der rohen Gradwerte wuerde den Mittelpunkt
        # faelschlich nahe 0 Grad statt nahe +-180 Grad landen lassen.
        positions = [{"id": "p1", "l": _entry([100.0, 200.0], phase=[179.0, -179.0])}]

        data = _interpolate_channel_to_common_grid(positions, "l")

        mid_freq = (100.0 * 200.0) ** 0.5
        mid_idx = int(np.argmin(np.abs(data.freq_hz - mid_freq)))
        mid_phase = data.phase_deg[0, mid_idx]
        wrapped = ((mid_phase + 180.0) % 360.0) - 180.0
        assert abs(wrapped) > 150.0, (
            f"Phase am Mittelpunkt ist {mid_phase:.1f} Grad (gewrappt {wrapped:.1f}) - "
            "sollte nahe +-180 Grad liegen, nicht nahe 0 Grad (Wrap-Bug)"
        )


class TestBuildJointLrData:
    def test_raises_when_no_full_lr_pair(self):
        positions = [
            {"id": "only_l", "l": _entry(np.linspace(20, 2000, 50))},
            {"id": "only_r", "r": _entry(np.linspace(20, 2000, 50))},
        ]
        with pytest.raises(ValueError, match="L.R-Paar"):
            _build_joint_lr_data(positions)

    def test_raises_on_duplicate_id_among_pairs(self):
        entry = _entry(np.linspace(20, 2000, 50))
        positions = [
            {"id": "dup", "l": entry, "r": entry},
            {"id": "dup", "l": entry, "r": entry},
        ]
        with pytest.raises(ValueError, match="doppelte"):
            _build_joint_lr_data(positions)

    def test_raises_when_phase_missing_on_one_pair(self):
        freq = np.linspace(20, 2000, 50).tolist()
        full = _entry(freq)
        no_phase = {"freq_hz": freq, "spl_db": [70.0] * len(freq)}
        positions = [
            {"id": "p1", "l": full, "r": full},
            {"id": "p2", "l": no_phase, "r": full},
        ]
        with pytest.raises(ValueError, match="Phase"):
            _build_joint_lr_data(positions)

    def test_excludes_position_missing_one_channel(self):
        full = _entry(np.linspace(20, 2000, 50))
        positions = [
            {"id": "p1", "l": full, "r": full},
            {"id": "p2", "l": full, "r": full},
            {"id": "incomplete", "l": full},  # kein 'r'
        ]
        joint = _build_joint_lr_data(positions)
        assert joint.position_ids == ["p1", "p2"]

    def test_builds_intersection_of_frequency_ranges_and_sorted_ids(self):
        # L und R haben absichtlich UNTERSCHIEDLICHE Frequenzbereiche - das
        # gemeinsame Raster muss die Schnittmenge sein, nicht das Raster
        # eines einzelnen Kanals.
        l_freq = np.linspace(20.0, 500.0, 50)
        r_freq = np.linspace(100.0, 2000.0, 50)
        positions = [
            {"id": "b_position", "l": _entry(l_freq), "r": _entry(r_freq)},
            {"id": "a_position", "l": _entry(l_freq), "r": _entry(r_freq)},
        ]
        joint = _build_joint_lr_data(positions)

        assert joint.position_ids == ["a_position", "b_position"]  # nach ID sortiert
        assert joint.freq_hz.min() >= 100.0 - 1e-6
        assert joint.freq_hz.max() <= 500.0 + 1e-6
        assert joint.l_spl_db.shape == (2, joint.freq_hz.size)
        assert joint.r_spl_db.shape == (2, joint.freq_hz.size)

    def test_unwraps_phase_discontinuity(self):
        full_wrapped = _entry([100.0, 200.0], phase=[179.0, -179.0])
        positions = [{"id": "p1", "l": full_wrapped, "r": full_wrapped}]
        joint = _build_joint_lr_data(positions)
        mid_idx = int(np.argmin(np.abs(joint.freq_hz - (100.0 * 200.0) ** 0.5)))
        wrapped = ((joint.l_phase_deg[0, mid_idx] + 180.0) % 360.0) - 180.0
        assert abs(wrapped) > 150.0


class TestRunPipelineJointLrHardening:
    def test_duplicate_id_does_not_crash_and_notes_skip(self, tmp_path):
        positions = [_synthetic_position("sofa_mitte", with_ir=True)]
        positions += [_synthetic_position("wander_0"), _synthetic_position("wander_0")]
        data = {"room_dimensions_m": [5.72, 3.72, 2.40], "positions": positions}
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(data), encoding="utf-8")

        report = run_pipeline(input_path, tmp_path / "out")

        assert any(
            "Do-No-Harm-Check uebersprungen" in n and "doppelte" in n for n in report.notes
        )

    def test_mismatched_lr_frequency_ranges_still_yields_do_no_harm_verdict(self, tmp_path):
        def position_with_shifted_r(pid, r_start, r_end):
            pos = _synthetic_position(pid)
            freq_r = np.linspace(r_start, r_end, 400)
            spl_r = np.full_like(freq_r, 70.0)
            spl_r += 8.0 / (1.0 + ((freq_r - 60.0) / 3.0) ** 2)
            pos["r"] = {
                "freq_hz": freq_r.tolist(), "spl_db": spl_r.tolist(),
                "phase_deg": np.zeros_like(freq_r).tolist(),
            }
            return pos

        # L bleibt auf 20-2000 Hz, R ist bewusst auf ein ANDERES, nur
        # teilweise ueberlappendes Raster (100-1500 Hz) verschoben.
        positions = [position_with_shifted_r("sofa_mitte", 100.0, 1500.0)]
        positions += [position_with_shifted_r(f"wander_{i}", 100.0, 1500.0) for i in range(3)]
        data = {"room_dimensions_m": [5.72, 3.72, 2.40], "positions": positions}
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(data), encoding="utf-8")

        report = run_pipeline(input_path, tmp_path / "out")

        do_no_harm_notes = [n for n in report.notes if "Do-No-Harm" in n]
        assert do_no_harm_notes, "kein Do-No-Harm-Hinweis im Report"
        assert not any("uebersprungen" in n for n in do_no_harm_notes)

    def test_incomplete_lr_pair_is_excluded_not_fatal(self, tmp_path):
        positions = [_synthetic_position("sofa_mitte", with_ir=True)]
        positions += [_synthetic_position(f"wander_{i}") for i in range(2)]
        incomplete = _synthetic_position("incomplete")
        del incomplete["r"]
        positions.append(incomplete)
        data = {"room_dimensions_m": [5.72, 3.72, 2.40], "positions": positions}
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(data), encoding="utf-8")

        report = run_pipeline(input_path, tmp_path / "out")

        do_no_harm_notes = [n for n in report.notes if "Do-No-Harm" in n]
        assert any("bestanden" in n for n in do_no_harm_notes) or any(
            "vorgeschlagene Filter erhoehen" in n for n in do_no_harm_notes
        )
