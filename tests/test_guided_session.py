import hashlib
import json

import numpy as np
import pytest

from spatial_analyzer.guided_session import (
    BLOCKS,
    MANIFEST_VERSION,
    PROFILES,
    IdDetectionResult,
    ManifestReadError,
    OperatingState,
    OperatingStateError,
    PhaseEvaluationBlocked,
    QcThresholds,
    SessionManifest,
    SessionManifestEntry,
    SessionStep,
    TimingQcResult,
    TimingQcStatus,
    add_nachtrag,
    build_measurement_name,
    build_notes_text,
    build_parser,
    build_plan,
    check_aba_drift,
    check_clipping,
    check_pretrigger_noise,
    check_rew_metadata,
    detect_new_measurement,
    determine_timing_qc,
    load_manifest,
    main,
    parse_user_action,
    prompt_operating_state,
    prompt_setup_notes,
    prompt_smart453_preset,
    prompt_user_action,
    read_kef_snapshot,
    require_phase_evaluation,
    resolve_block,
    run_guided_session,
    validate_operating_state,
    write_notes_sidecar,
)
from spatial_analyzer.rew_client import ReadOnlyRewClient, RewApiError, RewMeasurementMeta


def _meta(uuid, title="", is_sweep=True):
    return RewMeasurementMeta(uuid=uuid, title=title, is_sweep=is_sweep, raw={})


class _FakeClient:
    """Minimaler Ersatz fuer ReadOnlyRewClient, der nur die drei vom Modul
    genutzten Methoden anbietet - absichtlich kein echter Netzwerk-Client."""

    def __init__(self, measurements, freq_responses=None, impulse_responses=None, raise_on_list=None):
        self._measurements = measurements
        self._freq_responses = freq_responses or {}
        self._impulse_responses = impulse_responses or {}
        self._raise_on_list = raise_on_list

    def list_measurements(self):
        if self._raise_on_list:
            raise self._raise_on_list
        return self._measurements

    def get_frequency_response(self, uuid):
        return self._freq_responses[uuid]

    def get_impulse_response_raw(self, uuid):
        return self._impulse_responses[uuid]

    def get_impulse_response(self, uuid):
        # t=0 weit hinter den kurzen Test-IRs, damit die 10-%-Regel greift.
        return {
            "samples": np.asarray(self._impulse_responses[uuid], dtype=np.float64),
            "start_time_s": -1.0,
            "sample_interval_s": 1e-3,
        }


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


class TestProfiles:
    def test_wohnzimmer_profile_has_11_positions_and_lr_channels(self):
        profile = PROFILES["wohnzimmer"]
        assert profile.position_ids == tuple(f"P{i:02d}" for i in range(1, 12))
        assert profile.channels == ("L", "R")
        assert profile.supports_aba is True

    def test_smart453_profile_has_cube_positions_and_no_aba(self):
        profile = PROFILES["smart453"]
        assert len(profile.position_ids) == 9
        assert all(pid.startswith("C") for pid in profile.position_ids)
        assert profile.supports_aba is False

    def test_build_plan_wohnzimmer_lr(self):
        profile = PROFILES["wohnzimmer"]
        plan = build_plan(profile, use_aba=False)
        assert len(plan) == 11 * 2
        assert plan[0] == SessionStep("P01", "L", None)
        assert plan[1] == SessionStep("P01", "R", None)

    def test_build_plan_wohnzimmer_aba(self):
        profile = PROFILES["wohnzimmer"]
        plan = build_plan(profile, use_aba=True)
        assert len(plan) == 11 * 3
        assert plan[0] == SessionStep("P01", None, "A1")
        assert plan[1] == SessionStep("P01", None, "B")
        assert plan[2] == SessionStep("P01", None, "A2")

    def test_build_plan_smart453_rejects_aba(self):
        profile = PROFILES["smart453"]
        with pytest.raises(ValueError):
            build_plan(profile, use_aba=True)

    def test_build_plan_smart453_mic_channel(self):
        profile = PROFILES["smart453"]
        plan = build_plan(profile, use_aba=False)
        assert len(plan) == 9
        assert plan[0] == SessionStep("C01", "MIC", None)


class TestBuildMeasurementName:
    def test_plain_channel_name(self):
        profile = PROFILES["wohnzimmer"]
        assert build_measurement_name(profile, "P03", channel="L") == "P03_L"

    def test_aba_name_has_prefix(self):
        profile = PROFILES["wohnzimmer"]
        assert build_measurement_name(profile, "P03", aba_state="A1") == "ABA_A1_P03"
        assert build_measurement_name(profile, "P03", aba_state="B") == "ABA_B_P03"

    def test_aba_rejected_for_unsupported_profile(self):
        profile = PROFILES["smart453"]
        with pytest.raises(ValueError):
            build_measurement_name(profile, "C01", aba_state="A1")

    def test_unknown_aba_state_rejected(self):
        profile = PROFILES["wohnzimmer"]
        with pytest.raises(ValueError):
            build_measurement_name(profile, "P01", aba_state="X")

    def test_unknown_channel_rejected(self):
        profile = PROFILES["wohnzimmer"]
        with pytest.raises(ValueError):
            build_measurement_name(profile, "P01", channel="SUB")

    def test_position_only_name(self):
        profile = PROFILES["smart453"]
        assert build_measurement_name(profile, "C01") == "C01"


# ---------------------------------------------------------------------------
# Benutzer-Eingabe
# ---------------------------------------------------------------------------


class TestUserAction:
    @pytest.mark.parametrize(
        "raw,expected",
        [("", "confirm"), ("  ", "confirm"), ("r", "repeat"), ("R", "repeat"), ("s", "skip"), ("q", "quit")],
    )
    def test_parse_user_action(self, raw, expected):
        from spatial_analyzer.guided_session import UserAction

        assert parse_user_action(raw) == UserAction(expected)

    def test_parse_user_action_unknown_returns_none(self):
        assert parse_user_action("xyz") is None

    def test_prompt_user_action_retries_on_unknown_input(self):
        inputs = iter(["???", "q"])
        messages = []
        from spatial_analyzer.guided_session import UserAction

        action = prompt_user_action(input_fn=lambda _: next(inputs), print_fn=messages.append)
        assert action == UserAction.QUIT
        assert any("Unbekannte Eingabe" in m for m in messages)


# ---------------------------------------------------------------------------
# ID-Diff-Pruefung (0, 1, > 1 neue Messungen)
# ---------------------------------------------------------------------------


class TestDetectNewMeasurement:
    def test_zero_new_measurements(self):
        known = {"u1", "u2"}
        current = [_meta("u1"), _meta("u2")]
        result = detect_new_measurement(known, current)
        assert result.status == "none_found"
        assert result.new_measurements == []

    def test_exactly_one_new_measurement(self):
        known = {"u1"}
        current = [_meta("u1"), _meta("u2")]
        result = detect_new_measurement(known, current)
        assert result.status == "ok"
        assert [m.uuid for m in result.new_measurements] == ["u2"]

    def test_more_than_one_new_measurement(self):
        known = {"u1"}
        current = [_meta("u1"), _meta("u2"), _meta("u3")]
        result = detect_new_measurement(known, current)
        assert result.status == "multiple_found"
        assert len(result.new_measurements) == 2


# ---------------------------------------------------------------------------
# QC: Clipping
# ---------------------------------------------------------------------------


class TestCheckClipping:
    def test_quiet_signal_is_ok(self):
        samples = np.full(100, 0.1)
        finding = check_clipping(samples, QcThresholds())
        assert finding.severity == "ok"

    def test_near_full_scale_warns(self):
        samples = np.full(100, 0.95)  # ~ -0.45 dBFS, innerhalb 1 dB Marge
        finding = check_clipping(samples, QcThresholds(clip_margin_dbfs=1.0))
        assert finding.severity == "warning"
        assert "clipping" == finding.check

    def test_empty_array_warns(self):
        finding = check_clipping(np.array([]), QcThresholds())
        assert finding.severity == "warning"

    def test_all_zero_warns(self):
        finding = check_clipping(np.zeros(10), QcThresholds())
        assert finding.severity == "warning"


# ---------------------------------------------------------------------------
# QC: Rauschboden/SNR
# ---------------------------------------------------------------------------


class TestCheckPretriggerNoise:
    def test_clean_signal_is_ok(self):
        rng = np.random.default_rng(0)
        samples = np.zeros(1000)
        samples[:100] = rng.normal(0, 1e-5, 100)  # sehr leiser Pre-Trigger
        samples[500] = 1.0  # deutlicher Peak
        finding = check_pretrigger_noise(samples, QcThresholds())
        assert finding.severity == "ok"

    def test_noisy_pretrigger_warns(self):
        samples = np.zeros(1000)
        samples[:100] = 0.5  # lauter "Rauschboden"
        samples[500] = 0.6  # Peak kaum ueber dem Rauschen
        finding = check_pretrigger_noise(samples, QcThresholds(min_pretrigger_snr_db=20.0))
        assert finding.severity == "warning"

    def test_too_few_samples_warns(self):
        finding = check_pretrigger_noise(np.array([1.0, 2.0]), QcThresholds())
        assert finding.severity == "warning"


# ---------------------------------------------------------------------------
# QC: A1/A2-Drift
# ---------------------------------------------------------------------------


class TestCheckAbaDrift:

    def test_pretrigger_end_caps_noise_segment_before_peak(self):
        # Peak liegt innerhalb der ersten 10 %: ohne Begrenzung landet er im
        # Rauschsegment (SNR ~0 dB), mit pretrigger_end davor nicht.
        rng = np.random.default_rng(0)
        samples = rng.normal(0.0, 1e-4, 1000)
        samples[50] = 1.0
        assert check_pretrigger_noise(samples, QcThresholds()).severity == "warning"
        assert check_pretrigger_noise(samples, QcThresholds(), pretrigger_end=40).severity == "ok"

    def test_pretrigger_end_negative_gives_warning(self):
        samples = np.concatenate([np.full(100, 1e-6), [1.0]])
        finding = check_pretrigger_noise(samples, QcThresholds(), pretrigger_end=-5)
        assert finding.severity == "warning"
    def test_identical_curves_are_ok(self):
        freqs = np.linspace(20, 200, 50)
        spl = np.full(50, 70.0)
        finding = check_aba_drift(spl, spl, freqs, QcThresholds())
        assert finding.severity == "ok"

    def test_large_drift_warns(self):
        freqs = np.linspace(20, 200, 50)
        a1 = np.full(50, 70.0)
        a2 = np.full(50, 73.0)  # 3 dB Drift, ueber Default-Schwelle 1 dB
        finding = check_aba_drift(a1, a2, freqs, QcThresholds(aba_drift_max_db=1.0))
        assert finding.severity == "warning"
        assert "aba_drift" == finding.check

    def test_drift_outside_band_is_ignored(self):
        freqs = np.linspace(200, 400, 50)  # komplett ausserhalb 20-120 Hz
        a1 = np.full(50, 70.0)
        a2 = np.full(50, 90.0)
        finding = check_aba_drift(a1, a2, freqs, QcThresholds())
        assert finding.severity == "warning"
        assert "keine Frequenzpunkte" in finding.message

    def test_mismatched_shapes_warns(self):
        finding = check_aba_drift(np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), QcThresholds())
        assert finding.severity == "warning"


# ---------------------------------------------------------------------------
# aiokef-Snapshot + Offline-Fallback
# ---------------------------------------------------------------------------


class TestReadKefSnapshot:
    def test_successful_snapshot_via_injected_factory(self):
        class _Mode:
            desk_mode = "0dB"
            wall_mode = "0dB"
            phase_correction = True
            high_pass = "90Hz"
            sub_polarity = "+"
            bass_extension = "Standard"

        class _FakeSpeaker:
            async def get_mode(self):
                return _Mode()

        snapshot = read_kef_snapshot("192.168.1.50", client_factory=lambda host: _FakeSpeaker())
        assert snapshot == {
            "desk_mode": "0dB",
            "wall_mode": "0dB",
            "phase_correction": True,
            "high_pass": "90Hz",
            "sub_polarity": "+",
            "bass_extension": "Standard",
        }

    def test_missing_aiokef_falls_back_to_none(self):
        messages = []

        def factory(host):
            raise ImportError("aiokef nicht installiert")

        result = read_kef_snapshot("192.168.1.50", client_factory=factory, print_fn=messages.append)
        assert result is None
        assert any("aiokef" in m for m in messages)

    def test_network_error_falls_back_to_none(self):
        messages = []

        class _BrokenSpeaker:
            async def get_mode(self):
                raise OSError("device unreachable")

        result = read_kef_snapshot(
            "192.168.1.50", client_factory=lambda host: _BrokenSpeaker(), print_fn=messages.append
        )
        assert result is None
        assert any("fehlgeschlagen" in m for m in messages)


class TestPromptSmart453Preset:
    def test_returns_stripped_input(self):
        assert prompt_smart453_preset(input_fn=lambda _: "  Preset A  \n") == "Preset A"


# ---------------------------------------------------------------------------
# Notes-Text & lokale Sidecar-Datei (kein REW-API-Write)
# ---------------------------------------------------------------------------


class TestNotesTextAndSidecar:
    def test_notes_text_contains_core_fields(self):
        text = build_notes_text(
            "P01",
            "L",
            None,
            kef_snapshot={"wall_mode": "0dB"},
            smart453_preset=None,
            port_state="stopfen_ein",
            signal_path="USB",
        )
        assert "Position: P01" in text
        assert "Kanal: L" in text
        assert "KEF wall_mode: 0dB" in text
        assert "Messweg: USB" in text

    def test_notes_text_marks_missing_kef_snapshot(self):
        text = build_notes_text("P01", "L", None, None, None, None, None)
        assert "nicht verfuegbar" in text

    def test_write_notes_sidecar_creates_file(self, tmp_path):
        path = write_notes_sidecar(tmp_path / "notes", "P01_L", "Position: P01")
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "Position: P01"

    def test_read_only_client_has_no_notes_write_method(self):
        # Architektur-Invariante dieses PRs: rew_client.py bleibt GET-only,
        # kein Notes-Write ueber die REW-API (siehe Modul-Docstring).
        client = ReadOnlyRewClient()
        public_methods = [m for m in dir(client) if not m.startswith("_")]
        assert not any("notes" in m.lower() for m in public_methods)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestSessionManifest:
    def test_write_and_reload(self, tmp_path):
        manifest = SessionManifest(profile="wohnzimmer", dry_run=True, created_at="2026-11-23T00:00:00+00:00")
        manifest.add(
            SessionManifestEntry(
                measurement_name="P01_L",
                position_id="P01",
                channel="L",
                aba_state=None,
                uuid="u1",
                detection_status="ok",
                qc_findings=[{"severity": "ok", "check": "clipping", "message": "..."}],
                kef_snapshot=None,
                smart453_preset=None,
                notes_text="Position: P01",
                timestamp="2026-11-23T00:00:01+00:00",
            )
        )
        path = tmp_path / "manifest.json"
        manifest.write(path)

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["profile"] == "wohnzimmer"
        assert len(loaded["entries"]) == 1
        assert loaded["entries"][0]["measurement_name"] == "P01_L"


# ---------------------------------------------------------------------------
# --dry-run (End-zu-End ueber run_guided_session)
# ---------------------------------------------------------------------------


class TestRunGuidedSessionDryRun:
    def test_dry_run_completes_all_steps_without_real_client_calls(self, tmp_path):
        profile = PROFILES["wohnzimmer"]
        client = _FakeClient(measurements=[], raise_on_list=RewApiError("sollte nicht aufgerufen werden"))

        manifest = run_guided_session(
            profile,
            use_aba=False,
            client=client,
            manifest_path=tmp_path / "manifest.json",
            notes_dir=tmp_path / "notes",
            kef_host=None,
            dry_run=True,
            input_fn=lambda _: "",
            print_fn=lambda _: None,
        )

        assert len(manifest.entries) == 11 * 2
        assert all(e.detection_status == "none_found" for e in manifest.entries)
        assert (tmp_path / "manifest.json").exists()

    def test_dry_run_quit_is_not_triggered_by_confirm_simulation(self, tmp_path):
        # Im --dry-run-Modus wird CONFIRM simuliert, nicht REPEAT/QUIT - die
        # Session muss daher trotz einer rein zur Deko uebergebenen 'q'-
        # Eingabefunktion komplett durchlaufen.
        profile = PROFILES["smart453"]
        client = _FakeClient(measurements=[])

        manifest = run_guided_session(
            profile,
            use_aba=False,
            client=client,
            manifest_path=tmp_path / "manifest.json",
            notes_dir=tmp_path / "notes",
            kef_host=None,
            dry_run=True,
            input_fn=lambda _: "q",
            print_fn=lambda _: None,
        )
        assert len(manifest.entries) == 9


# ---------------------------------------------------------------------------
# run_guided_session mit Fake-Client: ID-Erkennung, QC, A1/A2-Drift, Quit
# ---------------------------------------------------------------------------


class TestRunGuidedSessionWithFakeClient:
    def test_quit_stops_early_and_saves_manifest(self, tmp_path):
        profile = PROFILES["wohnzimmer"]
        client = _FakeClient(measurements=[])
        inputs = iter(["q"])

        manifest = run_guided_session(
            profile,
            use_aba=False,
            client=client,
            manifest_path=tmp_path / "manifest.json",
            notes_dir=tmp_path / "notes",
            kef_host=None,
            dry_run=False,
            input_fn=lambda _: next(inputs),
            print_fn=lambda _: None,
        )
        assert len(manifest.entries) == 0
        assert (tmp_path / "manifest.json").exists()

    def test_skip_is_logged_as_skipped_entry_without_detection(self, tmp_path):
        profile = PROFILES["wohnzimmer"]
        client = _FakeClient(measurements=[])
        # 22 Schritte (11 Positionen x L/R) - jeden ueberspringen, dann quit
        inputs = iter(["s"] * 22)

        manifest = run_guided_session(
            profile,
            use_aba=False,
            client=client,
            manifest_path=tmp_path / "manifest.json",
            notes_dir=tmp_path / "notes",
            kef_host=None,
            dry_run=False,
            input_fn=lambda _: next(inputs),
            print_fn=lambda _: None,
        )
        assert len(manifest.entries) == 22
        assert all(e.status == "skipped" for e in manifest.entries)
        assert all(e.uuid is None and e.detection_status is None for e in manifest.entries)

    def test_confirm_with_clean_measurement_detected_and_logged(self, tmp_path):
        profile = PROFILES["wohnzimmer"]
        freqs = np.linspace(20, 200, 10)
        ir = np.concatenate([np.full(10, 1e-6), [0.5], np.zeros(20)])
        client = _FakeClient(
            measurements=[],
            freq_responses={"u1": {"freq_hz": freqs.tolist(), "spl_db": np.full(10, 70.0).tolist()}},
            impulse_responses={"u1": ir},
        )
        # Baseline (vor dem ersten Sweep) ist leer, erst nach der
        # Bestaetigung "erscheint" u1 in REW - genau das soll als neue
        # Messung erkannt werden.
        responses = [[], [_meta("u1")]]
        call_log = {"n": 0}

        def list_measurements():
            idx = min(call_log["n"], len(responses) - 1)
            call_log["n"] += 1
            return responses[idx]

        client.list_measurements = list_measurements

        # Erster Schritt P01_L bestaetigen, dann sofort beenden.
        inputs = iter(["", "q"])

        manifest = run_guided_session(
            profile,
            use_aba=False,
            client=client,
            manifest_path=tmp_path / "manifest.json",
            notes_dir=tmp_path / "notes",
            kef_host=None,
            dry_run=False,
            input_fn=lambda _: next(inputs),
            print_fn=lambda _: None,
        )
        assert len(manifest.entries) == 1
        entry = manifest.entries[0]
        assert entry.measurement_name == "P01_L"
        assert entry.uuid == "u1"
        assert entry.detection_status == "ok"
        clipping = [f for f in entry.qc_findings if f["check"] == "clipping"]
        assert len(clipping) == 1
        # REW normiert die IR auf Peak 1.0 -> Clipping ist nicht pruefbar
        # und darf nicht als "ok" erscheinen.
        assert clipping[0]["severity"] == "warning"
        assert "nicht pruefbar" in clipping[0]["message"]
        assert any(f["check"] == "noise_floor" for f in entry.qc_findings)
        assert (tmp_path / "notes" / "P01_L.txt").exists()

    def test_multiple_new_measurements_detected_warns_and_skips_qc(self, tmp_path):
        profile = PROFILES["wohnzimmer"]
        client = _FakeClient(measurements=[])
        # Baseline ist leer; nach der Bestaetigung "erscheinen" zwei neue
        # Messungen gleichzeitig (z. B. versehentlich doppelt ausgeloester
        # Sweep) - das muss als "multiple_found" erkannt werden.
        responses = [[], [_meta("u1"), _meta("u2")]]
        call_log = {"n": 0}

        def list_measurements():
            idx = min(call_log["n"], len(responses) - 1)
            call_log["n"] += 1
            return responses[idx]

        client.list_measurements = list_measurements
        messages = []
        inputs = iter(["", "q"])

        manifest = run_guided_session(
            profile,
            use_aba=False,
            client=client,
            manifest_path=tmp_path / "manifest.json",
            notes_dir=tmp_path / "notes",
            kef_host=None,
            dry_run=False,
            input_fn=lambda _: next(inputs),
            print_fn=messages.append,
        )
        assert manifest.entries[0].detection_status == "multiple_found"
        assert manifest.entries[0].uuid is None
        assert any("neue Messungen erkannt" in m for m in messages)

    def test_aba_a1_a2_drift_alarm_is_attached_to_a2_entry(self, tmp_path):
        profile = PROFILES["wohnzimmer"]
        freqs = np.linspace(20, 200, 10)
        a1_spl = np.full(10, 70.0)
        a2_spl = np.full(10, 75.0)  # 5 dB Drift im Band 20-120 Hz
        quiet_ir = np.concatenate([np.full(5, 1e-6), [0.1], np.zeros(5)])

        client = _FakeClient(
            measurements=[],
            freq_responses={
                "a1": {"freq_hz": freqs.tolist(), "spl_db": a1_spl.tolist()},
                "a2": {"freq_hz": freqs.tolist(), "spl_db": a2_spl.tolist()},
            },
            impulse_responses={"a1": quiet_ir, "a2": quiet_ir},
        )

        # list_measurements() liefert nacheinander: [] (Baseline), [a1] (nach
        # bestaetigtem A1). B wird uebersprungen und ruft list_measurements()
        # nicht auf, danach [a1, a2] (nach bestaetigtem A2).
        call_log = {"n": 0}
        responses = [[], [_meta("a1")], [_meta("a1"), _meta("a2")]]

        def list_measurements():
            idx = min(call_log["n"], len(responses) - 1)
            call_log["n"] += 1
            return responses[idx]

        client.list_measurements = list_measurements

        # Schritte fuer P01: A1 (bestaetigen), B (ueberspringen), A2 (bestaetigen), dann quit.
        inputs = iter(["", "s", "", "q"])

        manifest = run_guided_session(
            profile,
            use_aba=True,
            client=client,
            manifest_path=tmp_path / "manifest.json",
            notes_dir=tmp_path / "notes",
            kef_host=None,
            dry_run=False,
            input_fn=lambda _: next(inputs),
            print_fn=lambda _: None,
        )

        a2_entries = [e for e in manifest.entries if e.aba_state == "A2"]
        assert len(a2_entries) == 1
        drift_findings = [f for f in a2_entries[0].qc_findings if f["check"] == "aba_drift"]
        assert len(drift_findings) == 1
        assert drift_findings[0]["severity"] == "warning"


# ---------------------------------------------------------------------------
# Auto-Save nach jedem Schritt & --resume
# ---------------------------------------------------------------------------


def _scripted_client(responses, uuids=("u1", "u2", "u3", "u4")):
    """Fake-Client, dessen `list_measurements()` nacheinander die Eintraege
    aus `responses` liefert (letzter Eintrag wiederholt sich), inkl. gueltiger
    Frequenzgang-/Impulsantwort-Daten fuer alle `uuids`."""
    freqs = np.linspace(20, 200, 10)
    ir = np.concatenate([np.full(10, 1e-6), [0.5], np.zeros(20)])
    client = _FakeClient(
        measurements=[],
        freq_responses={u: {"freq_hz": freqs.tolist(), "spl_db": np.full(10, 70.0).tolist()} for u in uuids},
        impulse_responses={u: ir for u in uuids},
    )
    state = {"n": 0}

    def list_measurements():
        idx = min(state["n"], len(responses) - 1)
        state["n"] += 1
        return [_meta(u) for u in responses[idx]]

    client.list_measurements = list_measurements
    return client


def _run(profile, client, path, tmp_path, inputs, *, use_aba=False, resume=None, messages=None, **kwargs):
    it = iter(inputs)
    return run_guided_session(
        profile,
        use_aba=use_aba,
        client=client,
        manifest_path=path,
        notes_dir=tmp_path / "notes",
        kef_host=None,
        dry_run=False,
        input_fn=lambda _: next(it),
        print_fn=(messages.append if messages is not None else (lambda _: None)),
        resume_manifest=resume,
        **kwargs,
    )


class TestAutoSave:
    def test_manifest_is_written_after_every_completed_or_skipped_step(self, tmp_path):
        path = tmp_path / "manifest.json"
        client = _scripted_client([[], ["u1"], ["u1", "u2"]])
        entries_seen_at_prompt = []
        script = iter(["", "s", "", "q"])

        def input_fn(_):
            # Vor jeder Eingabe: wie viele Eintraege stehen bereits auf der Platte?
            entries_seen_at_prompt.append(len(json.loads(path.read_text(encoding="utf-8"))["entries"]) if path.exists() else 0)
            return next(script)

        run_guided_session(
            PROFILES["wohnzimmer"],
            use_aba=False,
            client=client,
            manifest_path=path,
            notes_dir=tmp_path / "notes",
            kef_host=None,
            dry_run=False,
            input_fn=input_fn,
            print_fn=lambda _: None,
        )
        # Schritt 1 (confirm), Schritt 2 (skip), Schritt 3 (confirm), dann q:
        # direkt nach jedem Schritt ist der Stand bereits auf der Platte.
        assert entries_seen_at_prompt == [0, 1, 2, 3]

    def test_write_is_atomic_and_leaves_no_temp_file(self, tmp_path):
        manifest = SessionManifest(profile="wohnzimmer", dry_run=False, created_at="t0")
        path = tmp_path / "sub" / "manifest.json"
        manifest.write(path)
        assert path.exists()
        assert not (tmp_path / "sub" / "manifest.json.tmp").exists()
        assert json.loads(path.read_text(encoding="utf-8"))["profile"] == "wohnzimmer"

    def test_manifest_stores_mode_and_conditions_for_resume(self, tmp_path):
        path = tmp_path / "manifest.json"
        _run(
            PROFILES["wohnzimmer"], _scripted_client([[]]), path, tmp_path, ["q"],
            use_aba=True, port_state="offen", signal_path="Bluetooth",
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["use_aba"] is True
        assert data["port_state"] == "offen"
        assert data["signal_path"] == "Bluetooth"


class TestResume:
    def test_quit_after_step_two_then_resume_continues_at_step_three_without_duplicates(self, tmp_path):
        path = tmp_path / "manifest.json"
        profile = PROFILES["wohnzimmer"]
        # Session 1: Baseline [], nach Schritt 1 [u1], nach Schritt 2 [u1,u2].
        # Session 2: Baseline [u1,u2], nach Schritt 3 [u1,u2,u3].
        client = _scripted_client([[], ["u1"], ["u1", "u2"], ["u1", "u2"], ["u1", "u2", "u3"]])

        _run(profile, client, path, tmp_path, ["", "", "q"])
        first = load_manifest(path)
        assert [e.measurement_name for e in first.entries] == ["P01_L", "P01_R"]

        messages = []
        resumed = _run(profile, client, path, tmp_path, ["", "q"], resume=first, messages=messages)

        names = [e.measurement_name for e in resumed.entries]
        assert names == ["P01_L", "P01_R", "P02_L"]  # fortgesetzt bei Schritt 3, keine Duplikate
        assert resumed is first  # dieselbe, erweiterte Manifest-Instanz
        assert [e.uuid for e in resumed.entries] == ["u1", "u2", "u3"]
        assert any("2 von 22 Schritten bereits abgeschlossen" in m for m in messages)
        assert any("Fortsetzung bei: P02_L" in m for m in messages)
        assert not any("=== P01_L ===" in m or "=== P01_R ===" in m for m in messages)
        # Auch der fortgesetzte Stand liegt sofort auf der Platte.
        assert [e.measurement_name for e in load_manifest(path).entries] == names

    def test_resume_offers_previously_skipped_step_again(self, tmp_path):
        path = tmp_path / "manifest.json"
        profile = PROFILES["wohnzimmer"]
        client = _scripted_client([[], ["u1"]])

        _run(profile, client, path, tmp_path, ["s", "q"])
        first = load_manifest(path)
        assert [e.status for e in first.entries] == ["skipped"]

        messages = []
        _run(profile, client, path, tmp_path, ["q"], resume=first, messages=messages)
        assert any("0 von 22 Schritten bereits abgeschlossen" in m for m in messages)
        assert any("=== P01_L ===" in m for m in messages)  # wird erneut angeboten

    def test_resume_with_all_steps_completed_does_nothing_more(self, tmp_path):
        path = tmp_path / "manifest.json"
        profile = PROFILES["smart453"]
        # Vollstaendige Dry-Run-Session erzeugen, dann fortsetzen.
        run_guided_session(
            profile, use_aba=False, client=_scripted_client([[]]), manifest_path=path,
            notes_dir=tmp_path / "notes", kef_host=None, dry_run=True,
            input_fn=lambda _: "", print_fn=lambda _: None,
        )
        first = load_manifest(path)
        assert len(first.entries) == 9

        messages = []
        resumed = _run(profile, _scripted_client([[]]), path, tmp_path, [], resume=first, messages=messages)
        assert len(resumed.entries) == 9
        assert any("bereits abgeschlossen" in m for m in messages)

    def test_old_manifest_without_status_field_counts_as_completed(self, tmp_path):
        path = tmp_path / "old.json"
        path.write_text(
            json.dumps(
                {
                    "profile": "wohnzimmer",
                    "dry_run": False,
                    "created_at": "t0",
                    "entries": [
                        {
                            "measurement_name": "P01_L",
                            "position_id": "P01",
                            "channel": "L",
                            "aba_state": None,
                            "uuid": "u1",
                            "detection_status": "ok",
                            "qc_findings": [],
                            "kef_snapshot": None,
                            "smart453_preset": None,
                            "notes_text": "x",
                            "timestamp": "t1",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        loaded = load_manifest(path)
        assert loaded.entries[0].status == "completed"
        assert loaded.use_aba is False and loaded.port_state is None

    def test_main_resume_takes_profile_mode_and_conditions_from_manifest(self, tmp_path, capsys):
        path = tmp_path / "manifest.json"
        # Unterbrochene A-B-A-Session (Bluetooth, Port offen) ohne einen Schritt.
        _run(
            PROFILES["wohnzimmer"], _scripted_client([[]]), path, tmp_path, ["q"],
            use_aba=True, port_state="offen", signal_path="Bluetooth",
        )

        # Widerspruechliche CLI-Flags muessen von --resume uebersteuert werden.
        exit_code = main(
            ["--resume", str(path), "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus", "--dynamic-eq", "aus", "--profile", "smart453",
             "--port-state", "stopfen_ein", "--signal-path", "USB", "--notes-dir", str(tmp_path / "notes")]
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "Fortsetzung: 0 von 33 Schritten bereits abgeschlossen" in out

        loaded = load_manifest(path)
        assert loaded.profile == "wohnzimmer"
        assert loaded.use_aba is True
        assert loaded.port_state == "offen"
        assert loaded.signal_path == "Bluetooth"
        assert len(loaded.entries) == 33
        assert loaded.entries[0].measurement_name == "ABA_A1_P01"

    def test_main_resume_with_missing_manifest_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--resume", str(tmp_path / "gibt_es_nicht.json")])

    def test_main_resume_with_unknown_profile_exits(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"profile": "unbekannt", "entries": []}), encoding="utf-8")
        with pytest.raises(SystemExit):
            main(["--resume", str(path)])

    def test_main_resume_with_corrupt_json_exits(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{ nicht json", encoding="utf-8")
        with pytest.raises(SystemExit):
            main(["--resume", str(path)])


# ---------------------------------------------------------------------------
# Block-Auswahl (--block 1..4) & Setup-Notiz (--notes)
# ---------------------------------------------------------------------------

_BLOCK_EXPECTED = {
    1: {"steps": 22, "aba": False, "port": "stopfen_ein", "sweep": "20 Hz - 20.000 Hz", "first": "P01_L"},
    2: {"steps": 33, "aba": True, "port": "stopfen_ein", "sweep": "15 Hz - 500 Hz", "first": "ABA_A1_P01"},
    3: {"steps": 33, "aba": True, "port": "offen", "sweep": "15 Hz - 500 Hz", "first": "ABA_A1_P01"},
    4: {"steps": 22, "aba": False, "port": "offen", "sweep": "20 Hz - 20.000 Hz", "first": "P01_L"},
}


class TestBlocks:
    @pytest.mark.parametrize("block", [1, 2, 3, 4])
    def test_run_with_block_uses_plan_port_state_and_sweep_hint(self, tmp_path, block):
        exp = _BLOCK_EXPECTED[block]
        messages = []
        manifest = run_guided_session(
            PROFILES["wohnzimmer"],
            use_aba=exp["aba"],
            client=_FakeClient(measurements=[]),
            manifest_path=tmp_path / "m.json",
            notes_dir=tmp_path / "notes",
            kef_host=None,
            dry_run=True,
            port_state=exp["port"],
            block=block,
            input_fn=lambda _: "",
            print_fn=messages.append,
        )
        assert len(manifest.entries) == exp["steps"]
        assert manifest.entries[0].measurement_name == exp["first"]
        assert manifest.block == block
        assert manifest.port_state == exp["port"]
        assert any(f"Block {block}:" in m for m in messages)
        assert any("REW-Sweep-Hinweis: " + exp["sweep"] in m for m in messages)
        data = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
        assert data["block"] == block and data["port_state"] == exp["port"] and data["use_aba"] is exp["aba"]
        # Der Block steht in jedem Notes-Entwurf.
        assert f"Block: {block}" in manifest.entries[0].notes_text

    def test_blocks_table_matches_plan_in_raum_akustik_section_16(self):
        assert {n: (b.use_aba, b.port_state, b.sweep_hint) for n, b in BLOCKS.items()} == {
            1: (False, "stopfen_ein", "20 Hz - 20.000 Hz"),
            2: (True, "stopfen_ein", "15 Hz - 500 Hz"),
            3: (True, "offen", "15 Hz - 500 Hz"),
            4: (False, "offen", "20 Hz - 20.000 Hz"),
        }

    def test_resolve_block_rejects_unknown_block_and_other_profile(self):
        with pytest.raises(ValueError):
            resolve_block(PROFILES["wohnzimmer"], 5)
        with pytest.raises(ValueError):
            resolve_block(PROFILES["smart453"], 1)

    @pytest.mark.parametrize(
        "block,use_aba,port",
        [(1, True, "stopfen_ein"), (1, False, "offen"), (3, True, "stopfen_ein"), (2, False, "stopfen_ein")],
    )
    def test_run_rejects_settings_that_contradict_block(self, tmp_path, block, use_aba, port):
        with pytest.raises(ValueError):
            run_guided_session(
                PROFILES["wohnzimmer"],
                use_aba=use_aba,
                client=_FakeClient(measurements=[]),
                manifest_path=tmp_path / "m.json",
                notes_dir=tmp_path / "notes",
                kef_host=None,
                dry_run=True,
                port_state=port,
                block=block,
                input_fn=lambda _: "",
                print_fn=lambda _: None,
            )

    def test_without_block_behaviour_is_unchanged(self, tmp_path):
        manifest = run_guided_session(
            PROFILES["wohnzimmer"],
            use_aba=False,
            client=_FakeClient(measurements=[]),
            manifest_path=tmp_path / "m.json",
            notes_dir=tmp_path / "notes",
            kef_host=None,
            dry_run=True,
            input_fn=lambda _: "",
            print_fn=lambda _: None,
        )
        assert manifest.block is None and manifest.setup_notes is None
        assert "Block:" not in manifest.entries[0].notes_text
        assert "Setup-Notiz" not in manifest.entries[0].notes_text

    @pytest.mark.parametrize("block", [1, 2, 3, 4])
    def test_main_block_sets_mode_and_port_state(self, tmp_path, capsys, block):
        exp = _BLOCK_EXPECTED[block]
        path = tmp_path / "m.json"
        exit_code = main(
            ["--block", str(block), "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus", "--dynamic-eq", "aus", "--manifest", str(path), "--notes-dir", str(tmp_path / "n")]
        )
        assert exit_code == 0
        loaded = load_manifest(path)
        assert loaded.block == block
        assert loaded.port_state == exp["port"]
        assert loaded.use_aba is exp["aba"]
        assert len(loaded.entries) == exp["steps"]
        assert "REW-Sweep-Hinweis: " + exp["sweep"] in capsys.readouterr().out

    def test_main_block_accepts_matching_explicit_port_state(self, tmp_path):
        exit_code = main(
            ["--block", "3", "--port-state", "offen", "--aba", "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus", "--dynamic-eq", "aus",
             "--manifest", str(tmp_path / "m.json"), "--notes-dir", str(tmp_path / "n")]
        )
        assert exit_code == 0

    @pytest.mark.parametrize(
        "argv",
        [
            ["--block", "1", "--aba"],
            ["--block", "3", "--port-state", "stopfen_ein"],
            ["--block", "4", "--port-state=stopfen_ein"],
            ["--block", "1", "--profile", "smart453"],
            ["--block", "5"],
        ],
    )
    def test_main_rejects_contradicting_or_invalid_block_flags(self, tmp_path, argv):
        with pytest.raises(SystemExit):
            main([*argv, "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus", "--dynamic-eq", "aus", "--manifest", str(tmp_path / "m.json")])
        assert not (tmp_path / "m.json").exists()

    def test_parser_block_default_is_none(self):
        args = build_parser().parse_args([])
        assert args.block is None and args.notes is None


class TestSetupNotes:
    def test_build_notes_text_includes_block_and_setup_notes_only_when_given(self):
        text = build_notes_text("P01", "L", None, None, None, "stopfen_ein", "USB", 1, "Phasenkorrektur EIN, Saxx-Pegel 0 dB")
        assert "Block: 1" in text
        assert "Setup-Notiz: Phasenkorrektur EIN, Saxx-Pegel 0 dB" in text
        plain = build_notes_text("P01", "L", None, None, None, None, None)
        assert "Block:" not in plain and "Setup-Notiz" not in plain

    def test_prompt_setup_notes_enter_skips_and_text_is_stripped(self):
        assert prompt_setup_notes(lambda _: "") is None
        assert prompt_setup_notes(lambda _: "   ") is None
        assert prompt_setup_notes(lambda _: "  KEF HP 90 Hz  ") == "KEF HP 90 Hz"

    def test_notes_flag_is_saved_in_manifest_and_notes_sidecar(self, tmp_path):
        path = tmp_path / "m.json"
        note = "Phasenkorrektur AUS, KEF HP 90 Hz, Saxx LP 80 Hz, Phase 135, Pegelregler -3 dB"
        exit_code = main(
            ["--block", "2", "--notes", note, "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus", "--dynamic-eq", "aus", "--manifest", str(path), "--notes-dir", str(tmp_path / "n")]
        )
        assert exit_code == 0
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["setup_notes"] == note
        assert f"Setup-Notiz: {note}" in data["entries"][0]["notes_text"]
        sidecar = (tmp_path / "n" / "ABA_A1_P01.txt").read_text(encoding="utf-8")
        assert f"Setup-Notiz: {note}" in sidecar

    def test_interactive_start_asks_once_and_enter_skips(self, tmp_path):
        prompts = []
        script = iter(["", "q"])  # Notiz: Enter, danach sofort Quit

        def input_fn(prompt):
            prompts.append(prompt)
            return next(script)

        manifest = run_guided_session(
            PROFILES["wohnzimmer"],
            use_aba=False,
            client=_scripted_client([[]]),
            manifest_path=tmp_path / "m.json",
            notes_dir=tmp_path / "notes",
            kef_host=None,
            dry_run=False,
            input_fn=input_fn,
            print_fn=lambda _: None,
            ask_setup_notes=True,
        )
        assert manifest.setup_notes is None
        assert sum("Setup-Notiz" in p for p in prompts) == 1

    def test_interactive_start_stores_typed_note(self, tmp_path):
        manifest = _run(
            PROFILES["wohnzimmer"], _scripted_client([[]]), tmp_path / "m.json", tmp_path,
            ["Saxx-Pegel 0 dB", "q"], ask_setup_notes=True,
        )
        assert manifest.setup_notes == "Saxx-Pegel 0 dB"
        assert json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))["setup_notes"] == "Saxx-Pegel 0 dB"

    def test_given_notes_skip_the_prompt(self, tmp_path):
        # Nur ein Eintrag im Skript (Quit): ein Prompt wuerde StopIteration ausloesen.
        manifest = _run(
            PROFILES["wohnzimmer"], _scripted_client([[]]), tmp_path / "m.json", tmp_path,
            ["q"], setup_notes="vorab gesetzt", ask_setup_notes=True,
        )
        assert manifest.setup_notes == "vorab gesetzt"

    def test_old_manifest_without_block_and_notes_loads_with_defaults(self, tmp_path):
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"profile": "wohnzimmer", "entries": []}), encoding="utf-8")
        loaded = load_manifest(path)
        assert loaded.block is None and loaded.setup_notes is None

    def test_resume_keeps_block_and_notes_and_new_notes_replace_them(self, tmp_path, capsys):
        path = tmp_path / "m.json"
        _run(
            PROFILES["wohnzimmer"], _scripted_client([[]]), path, tmp_path, ["q"],
            use_aba=True, port_state="offen", block=3, setup_notes="Stand 1",
        )
        assert load_manifest(path).block == 3

        # Resume ohne --notes: Block und Notiz bleiben erhalten.
        assert main(["--resume", str(path), "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus", "--dynamic-eq", "aus", "--notes-dir", str(tmp_path / "n")]) == 0
        loaded = load_manifest(path)
        assert loaded.block == 3 and loaded.setup_notes == "Stand 1"
        assert loaded.port_state == "offen" and loaded.use_aba is True
        assert len(loaded.entries) == 33
        assert "Block: 3" in loaded.entries[0].notes_text
        assert "Setup-Notiz: Stand 1" in loaded.entries[0].notes_text
        assert "REW-Sweep-Hinweis: 15 Hz - 500 Hz" in capsys.readouterr().out

    def test_resume_with_new_notes_replaces_stored_note(self, tmp_path):
        path = tmp_path / "m.json"
        _run(
            PROFILES["wohnzimmer"], _scripted_client([[]]), path, tmp_path, ["q"],
            use_aba=True, port_state="stopfen_ein", block=2, setup_notes="alt",
        )
        main(["--resume", str(path), "--notes", "neu", "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus", "--dynamic-eq", "aus", "--notes-dir", str(tmp_path / "n")])
        assert load_manifest(path).setup_notes == "neu"

    def test_resume_continues_block_without_duplicates(self, tmp_path):
        path = tmp_path / "m.json"
        profile = PROFILES["wohnzimmer"]
        # Session 1: Block 1, zwei Schritte bestaetigt, dann Quit.
        _run(
            profile, _scripted_client([[], ["u1"], ["u1", "u2"]]), path, tmp_path, ["", "", "q"],
            port_state="stopfen_ein", block=1, setup_notes="Phasenkorrektur EIN",
        )
        assert [e.measurement_name for e in load_manifest(path).entries] == ["P01_L", "P01_R"]
        # Session 2 per --resume (Dry-Run): Rest von Block 1, keine Duplikate.
        main(["--resume", str(path), "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus", "--dynamic-eq", "aus", "--notes-dir", str(tmp_path / "n")])
        names = [e.measurement_name for e in load_manifest(path).entries]
        assert len(names) == 22 and len(set(names)) == 22
        assert names[:3] == ["P01_L", "P01_R", "P02_L"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_parser_resume_default_is_none(self):
        assert build_parser().parse_args([]).resume is None

    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.profile == "wohnzimmer"
        assert args.aba is False
        assert args.dry_run is False
        assert args.port_state == "stopfen_ein"
        assert args.signal_path == "USB"
        assert args.manifest == "session_manifest.json"

    def test_parser_accepts_smart453_profile(self):
        parser = build_parser()
        args = parser.parse_args(["--profile", "smart453"])
        assert args.profile == "smart453"

    def test_aba_with_smart453_is_rejected_by_main(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            main(["--profile", "smart453", "--aba"])

    def test_main_dry_run_end_to_end(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        exit_code = main(["--profile", "wohnzimmer", "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus", "--dynamic-eq", "aus", "--manifest", "m.json", "--notes-dir", "notes"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Manifest geschrieben" in captured.out
        assert (tmp_path / "m.json").exists()


def test_port_state_parser_accepts_teilstopfen():
    from spatial_analyzer.guided_session import build_parser

    args = build_parser().parse_args(["--port-state", "teilstopfen"])
    assert args.port_state == "teilstopfen"


def test_repeat_refreshes_known_rew_uuids_before_next_confirmation(tmp_path):
    client = _scripted_client([[], ["old"], ["old", "new"]], uuids=("old", "new"))
    manifest = _run(
        PROFILES["wohnzimmer"], client, tmp_path / "manifest.json", tmp_path,
        ["r", "", "q"],
    )
    assert len(manifest.entries) == 1
    assert manifest.entries[0].uuid == "new"
    assert manifest.entries[0].detection_status == "ok"


# ---------------------------------------------------------------------------
# Paket B (Rev. 26): Timing-QC, REW-Metadaten, Betriebszustand, Manifest-Version,
# Legacy-Fortsetzung, Nachtraege. Bestehende Tests oben bleiben unveraendert.
# ---------------------------------------------------------------------------


class TestTimingQc:
    def test_status_values_are_separate_from_no_harm(self):
        assert [s.value for s in TimingQcStatus] == ["OK", "INCONCLUSIVE"]

    @pytest.mark.parametrize("legacy", [True, False])
    def test_timing_is_never_ok_without_documented_field(self, legacy):
        result = determine_timing_qc(legacy=legacy)
        assert result.status is TimingQcStatus.INCONCLUSIVE
        assert result.reason

    def test_phase_evaluation_is_blocked_while_inconclusive(self):
        timing = determine_timing_qc(legacy=False)
        with pytest.raises(PhaseEvaluationBlocked):
            require_phase_evaluation(timing)

    def test_phase_evaluation_only_passes_with_ok(self):
        require_phase_evaluation(TimingQcResult(TimingQcStatus.OK, "nachgewiesen"))

    def test_legacy_timing_is_inconclusive_with_reason(self):
        result = determine_timing_qc(legacy=True)
        assert result.status is TimingQcStatus.INCONCLUSIVE
        assert "Legacy" in result.reason


class TestRewMetadataCheck:
    VALID = {"sampleRate": 48000.0, "micCalPath": "C:\\cal\\UMIK-1 90deg.txt"}

    def test_valid_metadata_has_no_findings(self):
        assert check_rew_metadata(dict(self.VALID)) == []

    def test_missing_fields_are_reported_not_filled(self):
        raw: dict = {}
        findings = check_rew_metadata(raw)
        assert {(f.field, f.problem) for f in findings} == {("sampleRate", "fehlend"), ("micCalPath", "fehlend")}
        assert raw == {}  # nichts ergaenzt

    def test_empty_cal_path_counts_as_missing(self):
        findings = check_rew_metadata({"sampleRate": 48000.0, "micCalPath": "   "})
        assert [(f.field, f.problem) for f in findings] == [("micCalPath", "fehlend")]

    def test_wrong_sample_rate_is_contradictory(self):
        findings = check_rew_metadata({"sampleRate": 44100.0, "micCalPath": "UMIK-1 90deg.txt"})
        assert [(f.field, f.problem) for f in findings] == [("sampleRate", "widerspruechlich")]

    def test_non_numeric_sample_rate_is_invalid(self):
        findings = check_rew_metadata({"sampleRate": "48000", "micCalPath": "UMIK-1 90deg.txt"})
        assert [(f.field, f.problem) for f in findings] == [("sampleRate", "ungueltig")]

    def test_cal_file_not_umik_is_contradictory(self):
        findings = check_rew_metadata({"sampleRate": 48000.0, "micCalPath": "other_mic_90deg.txt"})
        assert [(f.field, f.problem) for f in findings] == [("micCalPath", "widerspruechlich")]

    def test_umik_without_90_degree_is_contradictory(self):
        findings = check_rew_metadata({"sampleRate": 48000.0, "micCalPath": "UMIK-1 0deg.txt"})
        assert [(f.field, f.problem) for f in findings] == [("micCalPath", "widerspruechlich")]

    def test_messages_do_not_expose_cal_path(self):
        findings = check_rew_metadata({"sampleRate": 48000.0, "micCalPath": "C:\\Users\\privat\\x.txt"})
        assert all("privat" not in f.message for f in findings)

    def test_metadata_check_does_not_change_timing_status(self):
        before = determine_timing_qc(legacy=False)
        check_rew_metadata({"sampleRate": 48000.0, "micCalPath": "UMIK-1 90deg.txt"})
        assert determine_timing_qc(legacy=False) == before


class TestOperatingState:
    def test_valid_state_has_no_problems(self):
        assert validate_operating_state("5cm_keil", "aus", "aus") == []

    def test_missing_or_invalid_values_are_reported(self):
        problems = validate_operating_state("  ", "vielleicht", None)
        assert len(problems) == 3

    def test_high_active_adds_warning_line(self):
        state = OperatingState("5cm_keil", "aktiv", "high active")
        assert any(line.startswith("WARNUNG: Dynamic EQ") for line in state.notes_lines())

    def test_aus_has_no_warning_line(self):
        state = OperatingState("5cm_keil", "aus", "aus")
        assert not any("WARNUNG" in line for line in state.notes_lines())

    def test_given_values_are_used_without_prompting(self):
        def no_input(_):
            raise AssertionError("keine Abfrage erwartet")

        state = prompt_operating_state(no_input, lambda _: None, door_state="5cm_keil", saxx_lpf="aus", dynamic_eq="aus")
        assert state == OperatingState("5cm_keil", "aus", "aus")

    def test_missing_values_are_prompted(self):
        answers = iter(["5cm_keil", "aktiv", "high active"])
        state = prompt_operating_state(lambda _: next(answers), lambda _: None)
        assert state == OperatingState("5cm_keil", "aktiv", "high active")

    def test_three_invalid_answers_abort_without_default(self):
        answers = iter(["", "vielleicht", "nein"])
        with pytest.raises(OperatingStateError):
            prompt_operating_state(lambda _: next(answers), lambda _: None, door_state="5cm_keil")

    def test_notes_without_state_are_unchanged(self):
        text = build_notes_text("P01", "L", None, None, None, "stopfen_ein", "USB")
        assert "Dynamic EQ" not in text and "Tuer-Zustand" not in text

    def test_notes_with_state_contain_all_lines(self):
        state = OperatingState("5cm_keil", "aktiv", "high active")
        text = build_notes_text("P01", "L", None, None, None, "stopfen_ein", "USB", operating_state=state)
        assert "Tuer-Zustand: 5cm_keil" in text
        assert "Saxx-LPF: aktiv" in text
        assert "Dynamic EQ: high active" in text
        assert "WARNUNG" in text


class TestManifestVersioning:
    def test_new_manifest_is_current_version_and_has_run_id_after_session(self, tmp_path):
        manifest = _run(PROFILES["wohnzimmer"], _scripted_client([[]]), tmp_path / "m.json", tmp_path, ["q"])
        assert manifest.manifest_version == MANIFEST_VERSION
        assert manifest.run_id
        assert manifest.timing_qc["status"] == "INCONCLUSIVE"
        assert not manifest.is_legacy

    def test_legacy_file_without_version_loads_as_legacy_without_filled_fields(self, tmp_path):
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps({"profile": "wohnzimmer", "entries": []}), encoding="utf-8")
        loaded = load_manifest(path)
        assert loaded.is_legacy
        assert loaded.run_id is None
        assert loaded.operating_state is None
        assert loaded.timing_qc is None
        assert loaded.nachtraege == []

    def test_entries_without_metadata_findings_load_with_empty_list(self, tmp_path):
        path = tmp_path / "legacy.json"
        path.write_text(
            json.dumps({"profile": "wohnzimmer", "entries": [{"measurement_name": "P01_L", "position_id": "P01", "channel": "L", "aba_state": None, "uuid": "u1", "detection_status": "ok", "qc_findings": [], "kef_snapshot": None, "smart453_preset": None, "notes_text": "x", "timestamp": "t"}]}),
            encoding="utf-8",
        )
        loaded = load_manifest(path)
        assert loaded.entries[0].metadata_findings == []

    def test_paket_b_fields_round_trip(self, tmp_path):
        manifest = SessionManifest(
            profile="wohnzimmer",
            dry_run=True,
            created_at="t0",
            run_id="abc",
            operating_state=OperatingState("5cm_keil", "aus", "aus").to_dict(),
            timing_qc=determine_timing_qc(legacy=False).to_dict(),
        )
        path = tmp_path / "m.json"
        manifest.write(path)
        loaded = load_manifest(path)
        assert loaded.run_id == "abc"
        assert loaded.operating_state == {"door_state": "5cm_keil", "saxx_lpf": "aus", "dynamic_eq": "aus"}
        assert loaded.timing_qc["status"] == "INCONCLUSIVE"
        assert not loaded.is_legacy


class TestNachtrag:
    def _manifest_with_entry(self):
        manifest = SessionManifest(profile="wohnzimmer", dry_run=False, created_at="t0")
        manifest.add(
            SessionManifestEntry(
                measurement_name="P01_L", position_id="P01", channel="L", aba_state=None, uuid="u1",
                detection_status="ok", qc_findings=[], kef_snapshot=None, smart453_preset=None,
                notes_text="x", timestamp="t1",
            )
        )
        return manifest

    def test_nachtrag_is_separate_and_marked_and_entries_unchanged(self):
        manifest = self._manifest_with_entry()
        before = [e.to_dict() for e in manifest.entries]
        item = add_nachtrag(manifest, "P01_L", "Tuer war offen", timestamp="2026-11-26T10:00:00+00:00")
        assert item["source"] == "nachtrag"
        assert item["timestamp"] == "2026-11-26T10:00:00+00:00"
        assert manifest.nachtraege == [item]
        assert [e.to_dict() for e in manifest.entries] == before

    def test_nachtrag_for_unknown_measurement_is_refused(self):
        with pytest.raises(ValueError):
            add_nachtrag(self._manifest_with_entry(), "P99_X", "irgendwas")

    def test_empty_nachtrag_is_refused(self):
        with pytest.raises(ValueError):
            add_nachtrag(self._manifest_with_entry(), "P01_L", "   ")


def _legacy_manifest_file(tmp_path, name="legacy.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "profile": "wohnzimmer",
                "dry_run": False,
                "created_at": "t0",
                "entries": [
                    {
                        "measurement_name": "P01_L", "position_id": "P01", "channel": "L", "aba_state": None,
                        "uuid": "u1", "detection_status": "ok", "qc_findings": [], "kef_snapshot": None,
                        "smart453_preset": None, "notes_text": "historisch", "timestamp": "t1",
                        "status": "completed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _predecessor_for(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "manifest_version": 1,
        "entries_inherited": 1,
    }


class TestLegacyContinuation:
    def test_legacy_file_is_never_modified_and_continuation_is_separate(self, tmp_path):
        legacy_path = _legacy_manifest_file(tmp_path)
        legacy_bytes = legacy_path.read_bytes()
        new_path = tmp_path / "fortsetzung.json"
        client = _scripted_client([["u1"], ["u1", "u2"]])

        _run(
            PROFILES["wohnzimmer"], client, new_path, tmp_path, ["", "q"],
            resume=load_manifest(legacy_path), predecessor=_predecessor_for(legacy_path),
        )

        assert legacy_path.read_bytes() == legacy_bytes  # Legacy unveraendert
        continued = load_manifest(new_path)
        assert not continued.is_legacy
        assert continued.run_id
        assert continued.predecessor["sha256"] == hashlib.sha256(legacy_bytes).hexdigest()
        names = [e.measurement_name for e in continued.entries]
        assert names == ["P01_L", "P01_R"]  # historischer Eintrag uebernommen, nicht dupliziert
        assert continued.entries[0].notes_text == "historisch"
        assert continued.timing_qc["status"] == "INCONCLUSIVE"

    def test_legacy_continuation_refuses_existing_target(self, tmp_path):
        legacy_path = _legacy_manifest_file(tmp_path)
        existing = tmp_path / "existiert.json"
        existing.write_text("{}", encoding="utf-8")
        with pytest.raises(FileExistsError):
            _run(
                PROFILES["wohnzimmer"], _scripted_client([[]]), existing, tmp_path, ["q"],
                resume=load_manifest(legacy_path), predecessor=_predecessor_for(legacy_path),
            )
        assert existing.read_text(encoding="utf-8") == "{}"

    def test_legacy_continuation_refuses_legacy_path_as_target(self, tmp_path):
        legacy_path = _legacy_manifest_file(tmp_path)
        before = legacy_path.read_bytes()
        with pytest.raises(FileExistsError):
            _run(
                PROFILES["wohnzimmer"], _scripted_client([[]]), legacy_path, tmp_path, ["q"],
                resume=load_manifest(legacy_path), predecessor=_predecessor_for(legacy_path),
            )
        assert legacy_path.read_bytes() == before

    def test_legacy_continuation_requires_predecessor(self, tmp_path):
        legacy_path = _legacy_manifest_file(tmp_path)
        with pytest.raises(ValueError):
            _run(
                PROFILES["wohnzimmer"], _scripted_client([[]]), tmp_path / "neu.json", tmp_path, ["q"],
                resume=load_manifest(legacy_path),
            )

    def test_new_session_refuses_existing_target(self, tmp_path):
        existing = tmp_path / "vorhanden.json"
        existing.write_text("{}", encoding="utf-8")
        with pytest.raises(FileExistsError):
            _run(PROFILES["wohnzimmer"], _scripted_client([[]]), existing, tmp_path, ["q"])
        assert existing.read_text(encoding="utf-8") == "{}"


class TestPaketBSessionIntegration:
    def test_missing_rew_metadata_is_reported_and_timing_stays_inconclusive(self, tmp_path):
        messages = []
        manifest = _run(
            PROFILES["wohnzimmer"], _scripted_client([[], ["u1"]]), tmp_path / "m.json", tmp_path, ["", "q"],
            messages=messages,
        )
        fields = {(f["field"], f["problem"]) for f in manifest.entries[0].metadata_findings}
        assert ("sampleRate", "fehlend") in fields
        assert ("micCalPath", "fehlend") in fields
        assert any(m.startswith("METADATEN FEHLEND") for m in messages)
        assert manifest.timing_qc["status"] == "INCONCLUSIVE"

    def test_high_active_warns_and_state_is_inherited_into_all_notes(self, tmp_path):
        messages = []
        state = OperatingState("5cm_keil", "aktiv", "high active")
        manifest = _run(
            PROFILES["wohnzimmer"], _scripted_client([[], ["u1"]]), tmp_path / "m.json", tmp_path, ["", "q"],
            messages=messages, operating_state=state,
        )
        assert any(m.startswith("WARNUNG: Dynamic EQ ist aktiv") for m in messages)
        assert manifest.operating_state == state.to_dict()
        assert "Dynamic EQ: high active" in manifest.entries[0].notes_text
        assert "Tuer-Zustand: 5cm_keil" in manifest.entries[0].notes_text


class TestPaketBCli:
    def test_non_dry_run_without_operating_state_aborts_before_any_measurement(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _: "")
        target = tmp_path / "neu.json"
        assert main(["--manifest", str(target), "--notes-dir", str(tmp_path / "n")]) == 2
        assert not target.exists()
        assert "keine Messung gestartet" in capsys.readouterr().err

    def test_existing_target_is_refused_by_cli(self, tmp_path):
        existing = tmp_path / "vorhanden.json"
        existing.write_text("{}", encoding="utf-8")
        with pytest.raises(SystemExit):
            main(["--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus", "--dynamic-eq", "aus", "--manifest", str(existing), "--notes-dir", str(tmp_path / "n")])
        assert existing.read_text(encoding="utf-8") == "{}"

    def test_invalid_saxx_lpf_choice_is_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--saxx-lpf", "vielleicht"])

    def test_legacy_resume_dry_run_writes_only_separate_manifest(self, tmp_path, capsys):
        legacy_path = _legacy_manifest_file(tmp_path)
        before = legacy_path.read_bytes()
        assert main(["--resume", str(legacy_path), "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus", "--dynamic-eq", "aus", "--notes-dir", str(tmp_path / "n")]) == 0
        assert legacy_path.read_bytes() == before
        out = capsys.readouterr().out
        assert "nur lesend geladen" in out
        continued = [p for p in tmp_path.glob("legacy_fortsetzung_*.json")]
        assert len(continued) == 1
        loaded = load_manifest(continued[0])
        assert not loaded.is_legacy
        assert loaded.predecessor["path"] == str(legacy_path.resolve())

    def test_resume_of_current_manifest_rejects_conflicting_operating_state(self, tmp_path):
        path = tmp_path / "m.json"
        _run(
            PROFILES["wohnzimmer"], _scripted_client([[]]), path, tmp_path, ["q"],
            operating_state=OperatingState("5cm_keil", "aus", "aus"),
        )
        with pytest.raises(SystemExit):
            main(["--resume", str(path), "--dry-run", "--door-state", "andere_tuer", "--notes-dir", str(tmp_path / "n")])


def _v2_manifest_file(tmp_path, name="v2.json"):
    """Echte Version-2-Datei aus einer Session (ohne Eintraege) mit Betriebszustand."""
    path = tmp_path / name
    _run(
        PROFILES["wohnzimmer"], _scripted_client([[]]), path, tmp_path, ["q"],
        operating_state=OperatingState("5cm_keil", "aus", "aus"),
    )
    return path


def _write_mutated(tmp_path, source, updates, name="mut.json"):
    data = json.loads(source.read_text(encoding="utf-8"))
    for key, value in updates.items():
        if value is _REMOVE:
            data.pop(key, None)
        else:
            data[key] = value
    target = tmp_path / name
    target.write_text(json.dumps(data), encoding="utf-8")
    return target


_REMOVE = object()


class TestSchemaValidation:
    @pytest.mark.parametrize(
        "updates",
        [
            {"operating_state": {"door_state": 5, "saxx_lpf": "aus", "dynamic_eq": "aus"}},
            {"operating_state": {"door_state": "x", "saxx_lpf": "vielleicht", "dynamic_eq": "aus"}},
            {"operating_state": {"door_state": "x", "saxx_lpf": "aus", "dynamic_eq": "aus", "extra": 1}},
            {"operating_state": {"door_state": "x", "saxx_lpf": "aus"}},
            {"operating_state": {"door_state": "", "saxx_lpf": "aus", "dynamic_eq": "aus"}},
            {"operating_state": ["x"]},
            {"timing_qc": {"status": "FERTIG", "reason": "x"}},
            {"timing_qc": {"status": "OK", "reason": "x", "extra": 1}},
            {"timing_qc": ["x"]},
            {"run_id": ""},
            {"run_id": 7},
            {"run_id": _REMOVE},
            {"nachtraege": _REMOVE},
            {"nachtraege": [1, "x"]},
            {"nachtraege": [{"source": "fremd", "target": "P01_L", "text": "x", "timestamp": "t"}]},
            {"nachtraege": [{"source": "nachtrag", "target": "P01_L", "text": "x"}]},
            {"predecessor": {"path": "x", "sha256": "nicht-hex"}},
            {"predecessor": {"path": "x"}},
            {"predecessor": "x"},
            {"block": 3, "use_aba": True, "port_state": None},
            {"block": 3, "use_aba": False, "port_state": "offen"},
            {"block": True},
            {"block": 99},
            {"block": "x"},
            {"profile": "smart453", "block": 1, "use_aba": False, "port_state": "stopfen_ein"},
            {"profile": "smart453", "use_aba": True},
            {"port_state": "unbekannt"},
            {"signal_path": "Funk"},
            {"created_at": 5},
            {"dry_run": "nein"},
            {"use_aba": "ja"},
            {"dry_run": _REMOVE},
            {"setup_notes": 5},
        ],
    )
    def test_invalid_version2_fields_are_read_errors(self, tmp_path, updates):
        source = _v2_manifest_file(tmp_path)
        bad = _write_mutated(tmp_path, source, updates)
        with pytest.raises(ManifestReadError):
            load_manifest(bad)
        with pytest.raises(SystemExit):
            main(["--resume", str(bad), "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus",
                  "--dynamic-eq", "aus", "--notes-dir", str(tmp_path / "n")])

    @pytest.mark.parametrize(
        "updates",
        [
            {"run_id": "abc"},
            {"operating_state": {"door_state": "x", "saxx_lpf": "aus", "dynamic_eq": "aus"}},
            {"timing_qc": {"status": "INCONCLUSIVE", "reason": "x"}},
            {"nachtraege": [{"source": "nachtrag", "target": "P01_L", "text": "x", "timestamp": "t"}]},
            {"block": 3, "use_aba": False, "port_state": "offen"},
            {"use_aba": True, "profile": "smart453"},
        ],
    )
    def test_legacy_with_version2_fields_or_contradictions_is_read_error(self, tmp_path, updates):
        legacy = _legacy_manifest_file(tmp_path)
        bad = _write_mutated(tmp_path, legacy, updates, name="legacy_mut.json")
        with pytest.raises(ManifestReadError):
            load_manifest(bad)

    @pytest.mark.parametrize(
        "entry_updates",
        [
            {"status": "kaputt"},
            {"position_id": 1},
            {"measurement_name": ""},
            {"qc_findings": "x"},
            {"metadata_findings": {"a": 1}},
            {"kef_snapshot": "text"},
            {"channel": 5},
            {"uuid": ["u1"]},
        ],
    )
    def test_invalid_entry_fields_are_read_errors_in_both_versions(self, tmp_path, entry_updates):
        legacy = _legacy_manifest_file(tmp_path)
        data = json.loads(legacy.read_text(encoding="utf-8"))
        data["entries"][0].update(entry_updates)
        bad = tmp_path / "entry_mut.json"
        bad.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ManifestReadError):
            load_manifest(bad)

    def test_valid_version2_file_still_loads(self, tmp_path):
        loaded = load_manifest(_v2_manifest_file(tmp_path))
        assert not loaded.is_legacy
        assert loaded.operating_state == {"door_state": "5cm_keil", "saxx_lpf": "aus", "dynamic_eq": "aus"}

    def test_eof_in_operating_state_prompt_becomes_operating_state_error(self):
        def closed_stdin(_):
            raise EOFError

        with pytest.raises(OperatingStateError):
            prompt_operating_state(closed_stdin, lambda _: None)

    def test_closed_stdin_in_real_run_exits_2_without_session(self, tmp_path, monkeypatch):
        def closed(_):
            raise EOFError

        monkeypatch.setattr("builtins.input", closed)
        target = tmp_path / "eof.json"
        assert main(["--manifest", str(target), "--notes-dir", str(tmp_path / "n")]) == 2
        assert not target.exists()

    def test_closed_stdin_in_dry_run_exits_2_without_session(self, tmp_path, monkeypatch):
        def closed(_):
            raise EOFError

        monkeypatch.setattr("builtins.input", closed)
        target = tmp_path / "eof_dry.json"
        assert main(["--dry-run", "--manifest", str(target), "--notes-dir", str(tmp_path / "n")]) == 2
        assert not target.exists()


class TestPaketBCorrections:
    def test_v2_with_null_timing_qc_is_read_error_and_not_redetermined_on_resume(self, tmp_path):
        # Befund F: `timing_qc: null` in Version 2 wird nicht mehr akzeptiert und beim
        # --resume nicht stillschweigend neu bestimmt (vorher: Neubestimmung).
        path = tmp_path / "m.json"
        _run(PROFILES["wohnzimmer"], _scripted_client([[]]), path, tmp_path, ["q"])
        data = json.loads(path.read_text(encoding="utf-8"))
        data["timing_qc"] = None
        path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ManifestReadError):
            load_manifest(path)
        with pytest.raises(SystemExit):
            main(["--resume", str(path), "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus",
                  "--dynamic-eq", "aus", "--notes-dir", str(tmp_path / "n")])

    def test_v2_with_null_operating_state_loads_and_is_asked_again_on_resume(self, tmp_path):
        # Bewusste V2-Regel: operating_state null = unvollstaendige Session; beim --resume
        # wird der Betriebszustand verbindlich neu abgefragt.
        path = tmp_path / "m.json"
        _run(PROFILES["wohnzimmer"], _scripted_client([[]]), path, tmp_path, ["q"])
        data = json.loads(path.read_text(encoding="utf-8"))
        data["operating_state"] = None
        path.write_text(json.dumps(data), encoding="utf-8")

        loaded = load_manifest(path)
        assert not loaded.is_legacy and loaded.operating_state is None

        resumed = _run(PROFILES["wohnzimmer"], _scripted_client([[]]), path, tmp_path, ["q"],
                       resume=loaded, operating_state=OperatingState("5cm_keil", "aus", "aus"))
        assert resumed.operating_state == {"door_state": "5cm_keil", "saxx_lpf": "aus", "dynamic_eq": "aus"}

    def test_v2_null_operating_state_on_cli_resume_without_answers_is_refused(self, tmp_path, monkeypatch, capsys):
        # Ohne Angaben und mit geschlossener Eingabe darf ein Resume mit null-Betriebszustand nicht starten.
        path = tmp_path / "m.json"
        _run(PROFILES["wohnzimmer"], _scripted_client([[]]), path, tmp_path, ["q"])
        data = json.loads(path.read_text(encoding="utf-8"))
        data["operating_state"] = None
        path.write_text(json.dumps(data), encoding="utf-8")

        def closed(_):
            raise EOFError

        monkeypatch.setattr("builtins.input", closed)
        before = path.read_text(encoding="utf-8")
        assert main(["--resume", str(path), "--dry-run", "--notes-dir", str(tmp_path / "n")]) == 2
        assert path.read_text(encoding="utf-8") == before
        assert "keine Messung gestartet" in capsys.readouterr().err

    @pytest.mark.parametrize("missing_or_bad", ["float_1_0", "missing"])
    def test_v2_predecessor_manifest_version_must_be_integer_and_present(self, tmp_path, missing_or_bad):
        # Befund B: JSON 1.0 wurde wegen 1.0 == 1 stillschweigend akzeptiert; in Version 2 ist das Feld Pflicht.
        legacy = _legacy_manifest_file(tmp_path)
        predecessor = _predecessor_for(legacy)
        if missing_or_bad == "float_1_0":
            predecessor["manifest_version"] = 1.0
        else:
            predecessor.pop("manifest_version")
        source = _v2_manifest_file(tmp_path)
        bad = _write_mutated(tmp_path, source, {"predecessor": predecessor}, name="pred.json")
        with pytest.raises(ManifestReadError):
            load_manifest(bad)

    def test_v2_entries_is_mandatory(self, tmp_path):
        # Befund A: `entries` ist in Version 2 striktes Pflichtfeld.
        bad = _write_mutated(tmp_path, _v2_manifest_file(tmp_path), {"entries": _REMOVE}, name="noentries.json")
        with pytest.raises(ManifestReadError):
            load_manifest(bad)

    @pytest.mark.parametrize(
        "updates",
        [
            {"run_id": None},
            {"operating_state": None},
            {"timing_qc": None},
            {"predecessor": None},
            {"nachtraege": []},
        ],
    )
    def test_legacy_v2_only_fields_null_or_empty_are_rejected(self, tmp_path, updates):
        # Befund I: V2-only-Felder in Legacy werden auch bei null/leer zurueckgewiesen.
        bad = _write_mutated(tmp_path, _legacy_manifest_file(tmp_path), updates, name="legacy_null.json")
        with pytest.raises(ManifestReadError):
            load_manifest(bad)

    def test_legacy_block_without_companion_fields_is_not_a_contradiction(self, tmp_path):
        # Befund D: fehlende Block-Begleitfelder in Legacy sind kein Widerspruch.
        path = tmp_path / "legacy_block.json"
        path.write_text(json.dumps({"profile": "wohnzimmer", "block": 1, "entries": []}), encoding="utf-8")
        loaded = load_manifest(path)
        assert loaded.block == 1 and loaded.is_legacy

    @pytest.mark.parametrize(
        "bad_state",
        [
            ("5cm_keil", None, "aus"),
            ("5cm_keil", "aus", None),
            (None, "aus", "aus"),
            ("", "aus", "aus"),
            ("   ", "aus", "aus"),
            (5, "aus", "aus"),
            ("5cm_keil", "", "aus"),
            ("5cm_keil", "aus", ""),
            ("5cm_keil", "aus", "unbekannt"),
        ],
    )
    def test_operating_state_is_strict_at_function_level(self, bad_state):
        # Befunde E/H: direkte Konstruktion ohne CLI/Manifest wird ebenfalls streng geprueft.
        with pytest.raises(OperatingStateError):
            OperatingState(*bad_state)

    @pytest.mark.parametrize("bad_raw", [["sampleRate", 48000], "UMIK-1 90deg.txt", 5])
    def test_rew_metadata_wrong_input_type_is_rejected(self, bad_raw):
        # Befund C: Eingabetyp wird explizit geprueft.
        with pytest.raises(TypeError):
            check_rew_metadata(bad_raw)

    def test_rew_metadata_none_still_reports_missing_fields(self):
        assert {f.field for f in check_rew_metadata(None)} == {"sampleRate", "micCalPath"}

    @pytest.mark.parametrize("name", ["UMIK-2 90deg.txt", "umik-2_90.txt", "UMIK-10 90deg.txt"])
    def test_umik2_or_other_umik_does_not_pass_umik1_plausibility(self, name):
        # Befund K: UMIK-2 darf die UMIK-1-Pruefung nicht bestehen.
        findings = check_rew_metadata({"sampleRate": 48000.0, "micCalPath": name})
        assert [(f.field, f.problem) for f in findings] == [("micCalPath", "widerspruechlich")]

    @pytest.mark.parametrize(
        "finding",
        [
            "kein Objekt",
            {"field": "sampleRate", "problem": "fehlend"},
            {"field": "sampleRate", "problem": "fehlend", "message": "x", "extra": 1},
            {"field": 5, "problem": "fehlend", "message": "x"},
            {"field": "sampleRate", "problem": "kaputt", "message": "x"},
            {"field": "sampleRate", "problem": "fehlend", "message": None},
            {"field": "", "problem": "fehlend", "message": "x"},
        ],
    )
    def test_invalid_metadata_findings_items_are_read_errors(self, tmp_path, finding):
        # Befund L: jedes Element von metadata_findings wird strukturell validiert.
        path = tmp_path / "mf.json"
        path.write_text(json.dumps({"profile": "wohnzimmer", "entries": [self._entry(metadata_findings=[finding])]}),
                        encoding="utf-8")
        with pytest.raises(ManifestReadError):
            load_manifest(path)

    def test_resume_v2_stored_ok_without_proof_is_downgraded(self, tmp_path):
        path = tmp_path / "m.json"
        _run(PROFILES["wohnzimmer"], _scripted_client([[]]), path, tmp_path, ["q"])
        data = json.loads(path.read_text(encoding="utf-8"))
        data["timing_qc"] = {"status": "OK", "reason": "von Hand eingetragen"}
        path.write_text(json.dumps(data), encoding="utf-8")

        messages = []
        resumed = _run(PROFILES["wohnzimmer"], _scripted_client([[]]), path, tmp_path, ["q"],
                       resume=load_manifest(path), messages=messages)
        assert resumed.timing_qc["status"] == "INCONCLUSIVE"
        assert any("neu bestimmt" in m for m in messages)

    @pytest.mark.parametrize("bad_version", ["2", 99, 1.5, True, None, [2]])
    def test_invalid_manifest_version_is_a_read_error(self, tmp_path, bad_version):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"profile": "wohnzimmer", "manifest_version": bad_version, "entries": []}),
                        encoding="utf-8")
        with pytest.raises(ManifestReadError):
            load_manifest(path)

    def test_invalid_manifest_version_in_cli_is_reported_as_read_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"profile": "wohnzimmer", "manifest_version": "2", "entries": []}),
                        encoding="utf-8")
        with pytest.raises(SystemExit):
            main(["--resume", str(path), "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus", "--dynamic-eq", "aus", "--notes-dir", str(tmp_path / "n")])
        assert path.read_text(encoding="utf-8").count('"2"') == 1  # unveraendert

    @pytest.mark.parametrize(
        "content",
        [
            {"profile": "wohnzimmer", "entries": 5},
            {"profile": "wohnzimmer", "entries": [5]},
            {"profile": "wohnzimmer", "entries": ["x"]},
            {"profile": ["wohnzimmer"], "entries": []},
            {"profile": None, "entries": []},
        ],
    )
    def test_malformed_entries_or_profile_are_read_errors_in_cli(self, tmp_path, content):
        path = tmp_path / "kaputt.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        with pytest.raises(ManifestReadError):
            load_manifest(path)
        # CLI: klare Fehlermeldung ueber parser.error, kein Traceback (kein TypeError/AttributeError).
        with pytest.raises(SystemExit):
            main(["--resume", str(path), "--dry-run", "--notes-dir", str(tmp_path / "n")])

    @staticmethod
    def _entry(**overrides):
        base = {
            "measurement_name": "P01_L", "position_id": "P01", "channel": "L", "aba_state": None,
            "uuid": "u1", "detection_status": "ok", "qc_findings": [], "kef_snapshot": None,
            "smart453_preset": None, "notes_text": "x", "timestamp": "t1", "status": "completed",
        }
        base.update(overrides)
        return base

    @pytest.mark.parametrize("block_value", [99, "x", [1], True])
    def test_reproduced_invalid_block_is_read_error_not_traceback(self, tmp_path, block_value):
        path = tmp_path / "block.json"
        path.write_text(json.dumps({"profile": "wohnzimmer", "block": block_value, "entries": []}), encoding="utf-8")
        with pytest.raises(ManifestReadError):
            load_manifest(path)
        with pytest.raises(SystemExit):
            main(["--resume", str(path), "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus",
                  "--dynamic-eq", "aus", "--notes-dir", str(tmp_path / "n")])

    def test_block_inconsistent_with_use_aba_is_read_error(self, tmp_path):
        path = tmp_path / "block.json"
        # Block 1 ist eine L/R-Einzelmessung ohne ABA; use_aba=True widerspricht dem.
        path.write_text(json.dumps({"profile": "wohnzimmer", "block": 1, "use_aba": True,
                                    "port_state": "stopfen_ein", "entries": []}), encoding="utf-8")
        with pytest.raises(ManifestReadError):
            load_manifest(path)

    def test_reproduced_position_id_list_is_read_error_not_traceback(self, tmp_path):
        path = tmp_path / "pos.json"
        path.write_text(json.dumps({"profile": "wohnzimmer", "entries": [self._entry(position_id=[])]}),
                        encoding="utf-8")
        with pytest.raises(ManifestReadError):
            load_manifest(path)
        with pytest.raises(SystemExit):
            main(["--resume", str(path), "--dry-run", "--door-state", "5cm_keil", "--saxx-lpf", "aus",
                  "--dynamic-eq", "aus", "--notes-dir", str(tmp_path / "n")])

    @pytest.mark.parametrize(
        "overrides",
        [
            {"measurement_name": None},
            {"measurement_name": ["P01_L"]},
            {"channel": ["L"]},
            {"aba_state": {"x": 1}},
            {"uuid": 42},
            {"detection_status": []},
            {"notes_text": None},
            {"qc_findings": "kaputt"},
            {"qc_findings": [5]},
            {"metadata_findings": {"a": 1}},
            {"kef_snapshot": "text"},
        ],
    )
    def test_invalid_entry_field_types_are_read_errors(self, tmp_path, overrides):
        path = tmp_path / "entry.json"
        path.write_text(json.dumps({"profile": "wohnzimmer", "entries": [self._entry(**overrides)]}),
                        encoding="utf-8")
        with pytest.raises(ManifestReadError):
            load_manifest(path)

    def test_valid_manifest_with_block_and_entries_still_loads(self, tmp_path):
        path = tmp_path / "ok.json"
        path.write_text(json.dumps({"profile": "wohnzimmer", "block": 1, "use_aba": False,
                                    "port_state": "stopfen_ein", "entries": [self._entry()]}),
                        encoding="utf-8")
        loaded = load_manifest(path)
        assert loaded.block == 1 and loaded.entries[0].position_id == "P01"

    def test_dry_run_without_operating_state_is_refused_without_session(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _: "")
        target = tmp_path / "dry.json"
        assert main(["--dry-run", "--manifest", str(target), "--notes-dir", str(tmp_path / "n")]) == 2
        assert not target.exists()
        assert "keine Messung gestartet" in capsys.readouterr().err
