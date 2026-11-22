import json

import numpy as np
import pytest

from spatial_analyzer.pipeline import run_pipeline


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
