import json

import numpy as np
import pytest

from spatial_analyzer import cli
from spatial_analyzer.grid import R160_STEPS_PER_DECADE
from spatial_analyzer.pipeline import run_pipeline
from spatial_analyzer.rew_import import (
    IR_WINDOW_AFTER_S,
    IR_WINDOW_BEFORE_S,
    RewImportError,
    build_pipeline_input,
    import_from_manifest,
)
from spatial_analyzer.validator import has_errors, validate_file

FS = 48000.0


def _entry(position, channel, uuid, status="completed", aba_state=None, notes="KEF-Lautstaerke: 40"):
    return {
        "measurement_name": f"{position}_{channel}",
        "position_id": position,
        "channel": channel,
        "aba_state": aba_state,
        "uuid": uuid,
        "status": status,
        "notes_text": notes,
    }


def _manifest(entries, **overrides):
    manifest = {"manifest_version": 2, "profile": "wohnzimmer", "dry_run": False, "entries": entries}
    manifest.update(overrides)
    return manifest


def _lr_entries(positions=("P01", "P02", "P03")):
    return [_entry(p, ch, f"{p}-{ch}") for p in positions for ch in ("L", "R")]


class FakeRewClient:
    """Liefert dekodierte Antworten wie `ReadOnlyRewClient` (Format am echten
    REW V5.20.13 belegt): linearer, ungeglaetteter Frequenzgang, IR mit t=0
    bei Index -startTime/sampleInterval."""

    def __init__(self, smoothing="None", phase_delay_s=1e-4):
        self.smoothing = smoothing
        self.phase_delay_s = phase_delay_s
        self.ir_calls = []

    def get_frequency_response(self, uuid):
        freq = 15.0 + np.arange(8000) * 0.5  # 15 ... ~4015 Hz, linear
        seed = sum(map(ord, uuid))
        spl = 75.0 + 3.0 * np.sin(freq / 37.0 + seed) + 6.0 * np.exp(-(((freq - 46.0) / 4.0) ** 2))
        phase = -360.0 * freq * self.phase_delay_s
        wrapped = (phase + 180.0) % 360.0 - 180.0
        return {"freq_hz": freq, "spl_db": spl, "phase_deg": wrapped, "smoothing": self.smoothing, "unit": "SPL"}

    def get_impulse_response(self, uuid):
        self.ir_calls.append(uuid)
        n = int(5 * FS)
        t0 = int(1.0 * FS)
        rng = np.random.default_rng(0)
        samples = rng.normal(0.0, 1e-5, n)
        t = np.arange(n - t0) / FS
        samples[t0:] += np.exp(-t / 0.3) * np.sin(2 * np.pi * 46.0 * t)
        return {
            "samples": samples,
            "sample_interval_s": 1.0 / FS,
            "start_time_s": -1.0,
            "sample_rate_hz": FS,
            "timing_reference": "Keine Timing-Referenz",
            "unit": "percent",
        }


class TestBuildPipelineInput:
    def test_maps_positions_channels_and_reference(self):
        client = FakeRewClient()
        result = build_pipeline_input(_manifest(_lr_entries()), client, (5.72, 3.72, 2.40))
        ids = [p["id"] for p in result["positions"]]
        assert ids == ["sofa_mitte", "P02", "P03"]
        assert result["room_dimensions_m"] == [5.72, 3.72, 2.40]
        ref = result["positions"][0]
        assert set(ref) == {"id", "l", "r"}
        assert ref["l"]["notes"] == "KEF-Lautstaerke: 40"
        # IR nur an der Referenzposition L/R abgerufen.
        assert sorted(client.ir_calls) == ["P01-L", "P01-R"]
        assert "impulse_response" not in result["positions"][1]["l"]

    def test_frequency_grid_contains_full_r160_lattice(self):
        result = build_pipeline_input(_manifest(_lr_entries()), FakeRewClient())
        freq = np.asarray(result["positions"][1]["r"]["freq_hz"])
        steps = np.log10(freq) * R160_STEPS_PER_DECADE
        on_lattice = np.isclose(steps, np.round(steps), atol=1e-9)
        expected = np.arange(np.ceil(np.log10(15.0) * 160), np.floor(np.log10(15.0 + 7999 * 0.5) * 160) + 1)
        np.testing.assert_allclose(np.sort(steps[on_lattice]), expected, atol=1e-9)
        assert np.all(np.diff(freq) > 0)
        assert len(result["positions"][1]["r"]["spl_db"]) == freq.size
        assert len(result["positions"][1]["r"]["phase_deg"]) == freq.size

    def test_phase_is_unwrapped_before_interpolation(self):
        result = build_pipeline_input(_manifest(_lr_entries()), FakeRewClient(phase_delay_s=1e-4))
        channel = result["positions"][1]["l"]
        expected = -360.0 * np.asarray(channel["freq_hz"]) * 1e-4
        # Rohphase ist gewickelt; entpackt+interpoliert muss die Gerade herauskommen.
        np.testing.assert_allclose(np.asarray(channel["phase_deg"]) - expected, 0.0, atol=0.5)

    def test_reference_ir_is_windowed_around_t0(self):
        result = build_pipeline_input(_manifest(_lr_entries()), FakeRewClient())
        ir = result["positions"][0]["l"]["impulse_response"]
        assert ir["sample_interval_s"] == pytest.approx(1.0 / FS)
        expected = int(round(IR_WINDOW_BEFORE_S * FS)) + int(round(IR_WINDOW_AFTER_S * FS))
        assert len(ir["samples"]) == expected
        peak = int(np.argmax(np.abs(ir["samples"])))
        assert int(round(IR_WINDOW_BEFORE_S * FS)) <= peak < int(round((IR_WINDOW_BEFORE_S + 0.05) * FS))

    def test_skipped_and_aba_entries_are_ignored(self):
        entries = _lr_entries() + [
            _entry("P04", "L", "x", status="skipped"),
            _entry("P01", "L", "aba", aba_state="A1"),
        ]
        result = build_pipeline_input(_manifest(entries), FakeRewClient())
        assert [p["id"] for p in result["positions"]] == ["sofa_mitte", "P02", "P03"]

    def test_sub_everywhere_is_included(self):
        entries = _lr_entries() + [_entry(p, "SAXX", f"{p}-S") for p in ("P01", "P02", "P03")]
        result = build_pipeline_input(_manifest(entries), FakeRewClient())
        assert all("sub" in p for p in result["positions"])

    @pytest.mark.parametrize(
        "manifest, match",
        [
            (_manifest(_lr_entries(), dry_run=True), "Dry-Run"),
            (_manifest(_lr_entries(), manifest_version=1), "manifest_version"),
            (_manifest(_lr_entries(), profile="smart453"), "Profil"),
            (_manifest(_lr_entries()[:-1]), "P03: Kanal/Kanaele r fehlen"),
            (_manifest(_lr_entries() + [_entry("P02", "L", "dup")]), "doppelt"),
            (_manifest(_lr_entries(("P02", "P03"))), "Referenzposition P01"),
            (_manifest(_lr_entries() + [_entry("P01", "SAXX", "s1")]), "Sub nur an einem Teil"),
            (_manifest(_lr_entries() + [_entry("P02", "C", "c")]), "Kanal 'C' unbekannt"),
            (_manifest(_lr_entries() + [_entry("P04", "L", None)]), "ohne REW-UUID"),
            (_manifest([]), "keine abgeschlossenen"),
        ],
    )
    def test_rejects_ambiguous_or_incomplete_manifests(self, manifest, match):
        with pytest.raises(RewImportError, match=match):
            build_pipeline_input(manifest, FakeRewClient())

    def test_rejects_smoothed_frequency_response(self):
        with pytest.raises(RewImportError, match="geglaettet"):
            build_pipeline_input(_manifest(_lr_entries()), FakeRewClient(smoothing="Psy"))

    def test_fast_rotating_phase_is_refined_not_rejected(self):
        # 20 ms Restlaufzeit wie bei echten REW-Daten: Raster wird verdichtet,
        # jeder Phasenschritt < 90 Grad, Phase bleibt die korrekte Gerade.
        result = build_pipeline_input(_manifest(_lr_entries()), FakeRewClient(phase_delay_s=0.02))
        channel = result["positions"][1]["l"]
        freq = np.asarray(channel["freq_hz"])
        phase = np.asarray(channel["phase_deg"])
        assert np.all(np.abs(np.diff(phase)) < 90.0)
        np.testing.assert_allclose(phase - (-360.0 * freq * 0.02), 0.0, atol=1.0)
        # Erneutes Entpacken (wie in der Pipeline) aendert nichts.
        np.testing.assert_allclose(np.rad2deg(np.unwrap(np.deg2rad(phase))), phase, atol=1e-9)

    def test_pipeline_grid_values_match_direct_interpolation_of_raw(self):
        # Die Pipeline interpoliert auf R160; aus den Konverterdaten muss an
        # den R160-Punkten dasselbe herauskommen wie direkt aus den Rohdaten.
        from spatial_analyzer.grid import interpolate_to_grid, r160_grid

        client = FakeRewClient(phase_delay_s=0.02)
        raw = client.get_frequency_response("P02-L")
        result = build_pipeline_input(_manifest(_lr_entries()), client)
        channel = result["positions"][1]["l"]
        grid = r160_grid(min(channel["freq_hz"]), max(channel["freq_hz"]))
        raw_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(raw["phase_deg"])))
        np.testing.assert_allclose(
            interpolate_to_grid(np.asarray(channel["freq_hz"]), np.asarray(channel["spl_db"]), grid),
            interpolate_to_grid(raw["freq_hz"], raw["spl_db"], grid),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            interpolate_to_grid(np.asarray(channel["freq_hz"]), np.asarray(channel["phase_deg"]), grid),
            interpolate_to_grid(raw["freq_hz"], raw_unwrapped, grid),
            atol=1e-6,
        )


class TestEndToEnd:
    def test_import_output_runs_through_validator_and_pipeline(self, tmp_path):
        manifest_path = tmp_path / "session_manifest.json"
        manifest_path.write_text(json.dumps(_manifest(_lr_entries())), encoding="utf-8")
        out = tmp_path / "input.json"
        import_from_manifest(manifest_path, out, FakeRewClient(), (5.72, 3.72, 2.40))
        assert not has_errors(validate_file(out))
        report = run_pipeline(out, tmp_path / "run")
        assert not any("sofa_mitte" in n and "Keine Position" in n for n in report.notes)
        assert (tmp_path / "run").is_dir()

    def test_cli_import_rew(self, tmp_path, monkeypatch, capsys):
        import spatial_analyzer.rew_client as rew_client

        monkeypatch.setattr(rew_client, "ReadOnlyRewClient", lambda **kwargs: FakeRewClient())
        manifest_path = tmp_path / "m.json"
        manifest_path.write_text(json.dumps(_manifest(_lr_entries())), encoding="utf-8")
        out = tmp_path / "in.json"
        rc = cli.main(["import-rew", str(manifest_path), str(out), "--room", "5.72", "3.72", "2.40"])
        assert rc == 0
        assert "Positionen: 3" in capsys.readouterr().out
        assert out.exists()

    def test_cli_import_rew_reports_error(self, tmp_path, monkeypatch, capsys):
        import spatial_analyzer.rew_client as rew_client

        monkeypatch.setattr(rew_client, "ReadOnlyRewClient", lambda **kwargs: FakeRewClient())
        manifest_path = tmp_path / "m.json"
        manifest_path.write_text(json.dumps(_manifest(_lr_entries(), dry_run=True)), encoding="utf-8")
        rc = cli.main(["import-rew", str(manifest_path), str(tmp_path / "x.json")])
        assert rc == 1
        assert "Import nicht moeglich" in capsys.readouterr().out
