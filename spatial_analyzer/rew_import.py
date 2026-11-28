"""Konverter: guided-session-Manifest + REW-API -> Pipeline-Eingabe-JSON.

Schliesst die Luecke zwischen der geführten Messsitzung (`guided_session.py`,
schreibt `session_manifest.json` mit REW-UUID je Messung) und `pipeline.py`
(liest die dokumentierte JSON-Zwischenform, siehe dortiger Docstring).
Ein echter `.mdat`-Byte-Parser ist dafuer nicht noetig: die Daten kommen
ueber die schreibgeschuetzte REW-API (`rew_client.py`, Format am echten
REW V5.20.13 belegt). Bezug: docs/SPECIFICATION.md Abschnitt 3.A
(Datenimport & Validierung) und 3.B (Frequenzraster); der `.mdat`-
Byte-Parser (Abschnitt G.4, Issue #7) bleibt davon unberuehrt.

Festlegungen (Projektleiter, 28.11.2026):
- Benennung aus dem Code ist verbindlich (`P01_L`, `P01_R`, ...).
- `P01` (Sofamitte) wird zur Pipeline-Referenzposition `sofa_mitte`.
- Ein Saxx-Kanal (`SAXX`/`SUB`) wird optional als `sub` uebernommen.

Phase nur mit Timing-Nachweis: REW setzt t=0 je Messung auf den Peak, die
Phase enthaelt eine beliebige Restlaufzeit. Ohne `timing_qc.status == "OK"`
im Manifest (Paket B, `require_phase_evaluation`) ist der Laufzeitbezug
zwischen L, R und Sub undefiniert. Dann wird KEINE Phase ausgegeben; die
Pipeline meldet das Do-No-Harm-Gate INCONCLUSIVE und blockiert den Export.
Heute liefert `determine_timing_qc` immer INCONCLUSIVE.

Hinweis: das Profil `wohnzimmer` kennt derzeit nur L/R; `SAXX`/`SUB`
kommen erst vor, wenn die guided session um den Saxx-Kanal erweitert ist.

Strikt statt still: fehlende Kanaele, doppelte Messungen, Dry-Run- oder
Legacy-Manifeste und ein nur teilweise vorhandener Sub fuehren zu
`RewImportError`, nie zu stillschweigend ergaenzten oder verworfenen Daten.

Datenmenge: REW liefert ungeglaettete Frequenzgaenge linear (~65.000
Punkte). Der Konverter interpoliert sie mit derselben Funktion wie die
Pipeline (`interpolate_to_grid`, log-linear) auf das feste R160-Raster.
Da R160 ein festes Gitter (10^(n/160)) ist, trifft die spaetere Pipeline-
Interpolation exakt dieselben Gitterpunkte - das Ergebnis entspricht dem,
was die Pipeline aus den Rohdaten selbst berechnet haette. Die Phase wird
dafuer wie in der Pipeline VOR der Interpolation entpackt. Wo sie sich
zwischen R160-Punkten zu schnell dreht (Restlaufzeit), wird das Raster mit
Zwischenpunkten verdichtet (`_refine_for_phase`), damit das erneute
Entpacken in der Pipeline nichts veraendert.

Impulsantwort: nur fuer `sofa_mitte` L/R (nur dort nutzt die Pipeline sie,
Abklinganalyse) und auf ein Fenster um t=0 begrenzt
(`IR_WINDOW_BEFORE_S`/`IR_WINDOW_AFTER_S`, Arbeitswerte).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from spatial_analyzer.grid import interpolate_to_grid, r160_grid

REFERENCE_SOURCE_POSITION = "P01"
REFERENCE_POSITION_ID = "sofa_mitte"
CHANNEL_MAP = {"L": "l", "R": "r", "SAXX": "sub", "SUB": "sub"}
SUPPORTED_PROFILE = "wohnzimmer"
SUPPORTED_MANIFEST_VERSION = 2
# Arbeitswerte: Fenster der Referenz-IR um t=0 (Bass-Abklingen < 3 s).
IR_WINDOW_BEFORE_S = 0.5
IR_WINDOW_AFTER_S = 3.0
# Phase: hoechstens 90 Grad zwischen gelieferten Punkten (Abstand zu den
# 180 Grad, ab denen np.unwrap in der Pipeline eingreift).
MAX_PHASE_STEP_DEG = 90.0
MAX_REFINE_ROUNDS = 30


class RewImportError(ValueError):
    """Manifest oder REW-Daten erlauben keinen eindeutigen Import."""


class _RewClient(Protocol):
    def get_frequency_response(self, uuid: str) -> dict[str, Any]: ...

    def get_impulse_response(self, uuid: str) -> dict[str, Any]: ...


def _select_entries(manifest: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Gibt {position_id: {kanal: eintrag}} der verwertbaren Messungen zurueck."""
    if manifest.get("manifest_version") != SUPPORTED_MANIFEST_VERSION:
        raise RewImportError(
            f"manifest_version {manifest.get('manifest_version')!r} nicht unterstuetzt "
            f"(erwartet {SUPPORTED_MANIFEST_VERSION}; Legacy-Manifeste haben keine Timing-QC)"
        )
    if manifest.get("profile") != SUPPORTED_PROFILE:
        raise RewImportError(f"Profil {manifest.get('profile')!r} nicht unterstuetzt (nur {SUPPORTED_PROFILE!r})")
    if manifest.get("dry_run"):
        raise RewImportError("Dry-Run-Manifest enthaelt keine echten Messungen")

    selected: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in manifest.get("entries", []):
        if entry.get("status", "completed") != "completed" or entry.get("aba_state") is not None:
            continue
        channel = CHANNEL_MAP.get(str(entry.get("channel", "")).upper())
        if channel is None:
            raise RewImportError(
                f"{entry.get('measurement_name')!r}: Kanal {entry.get('channel')!r} unbekannt "
                f"(erlaubt: {', '.join(CHANNEL_MAP)})"
            )
        if not entry.get("uuid"):
            raise RewImportError(f"{entry.get('measurement_name')!r}: abgeschlossen, aber ohne REW-UUID")
        position = str(entry.get("position_id", ""))
        if not position:
            raise RewImportError(f"{entry.get('measurement_name')!r}: position_id fehlt")
        slot = selected.setdefault(position, {})
        if channel in slot:
            raise RewImportError(
                f"Position {position}: Kanal {channel} doppelt "
                f"({slot[channel].get('measurement_name')!r} und {entry.get('measurement_name')!r})"
            )
        slot[channel] = entry

    if not selected:
        if any(e.get("aba_state") is not None for e in manifest.get("entries", [])):
            raise RewImportError(
                "Manifest enthaelt nur A-B-A-Messungen (Vergleichsfolge) - keine Positionsmessungen"
            )
        raise RewImportError("Manifest enthaelt keine abgeschlossenen Messungen")
    if REFERENCE_SOURCE_POSITION not in selected:
        raise RewImportError(f"Referenzposition {REFERENCE_SOURCE_POSITION} (Sofamitte) fehlt")
    for position, channels in sorted(selected.items()):
        missing = [c for c in ("l", "r") if c not in channels]
        if missing:
            raise RewImportError(f"Position {position}: Kanal/Kanaele {', '.join(missing)} fehlen")
    with_sub = [p for p, ch in selected.items() if "sub" in ch]
    if with_sub and len(with_sub) != len(selected):
        without = sorted(set(selected) - set(with_sub))
        raise RewImportError(
            f"Sub nur an einem Teil der Positionen gemessen (fehlt an {', '.join(without)}); "
            "die Pipeline nutzt den Sub nur, wenn er ueberall vorliegt"
        )
    return selected


def timing_ok(manifest: dict[str, Any]) -> bool:
    timing = manifest.get("timing_qc")
    return isinstance(timing, dict) and timing.get("status") == "OK"


def _resample_channel(fr: dict[str, Any], name: str, with_phase: bool) -> dict[str, Any]:
    freq = np.asarray(fr["freq_hz"], dtype=np.float64)
    spl = np.asarray(fr["spl_db"], dtype=np.float64)
    if fr.get("smoothing") not in (None, "None"):
        raise RewImportError(f"{name}: Frequenzgang ist geglaettet ({fr.get('smoothing')!r})")
    # Rohendpunkte mitliefern: so ergibt sich in der Pipeline derselbe
    # gemeinsame Bereich (common_start/end) wie aus den Rohdaten.
    grid = np.unique(np.concatenate([[freq[0]], r160_grid(float(freq[0]), float(freq[-1])), [freq[-1]]]))
    phase = None
    if with_phase and fr.get("phase_deg") is not None:
        unwrapped = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(fr["phase_deg"], dtype=np.float64))))
        grid, phase = _refine_for_phase(freq, unwrapped, grid, name)
    channel: dict[str, Any] = {
        "freq_hz": grid.tolist(),
        "spl_db": interpolate_to_grid(freq, spl, grid).tolist(),
    }
    if phase is not None:
        channel["phase_deg"] = phase.tolist()
    return channel


def _refine_for_phase(
    freq: np.ndarray, unwrapped: np.ndarray, grid: np.ndarray, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Verdichtet das R160-Raster dort, wo sich die (entpackte) Phase
    zwischen Nachbarpunkten um >= MAX_PHASE_STEP_DEG dreht.

    Hintergrund: echte REW-Phasen enthalten eine Restlaufzeit (am echten REW
    bis ~200.000 Grad ueber 20 Hz-22 kHz). Die Pipeline entpackt jede
    gelieferte Phase erneut; bei Spruengen >= 180 Grad zwischen Punkten
    wuerde sie die Phase verfaelschen. Eingefuegt werden geometrische
    Mittelpunkte (Werte aus den Rohdaten, gleiche Interpolation wie die
    Pipeline). Alle R160-Gitterpunkte bleiben erhalten, die Pipeline trifft
    sie daher exakt."""
    phase = interpolate_to_grid(freq, unwrapped, grid)
    for _ in range(MAX_REFINE_ROUNDS):
        steps = np.abs(np.diff(phase))
        too_wide = np.nonzero(steps >= MAX_PHASE_STEP_DEG)[0]
        if too_wide.size == 0:
            return grid, phase
        mids = np.sqrt(grid[too_wide] * grid[too_wide + 1])
        grid = np.sort(np.concatenate([grid, mids]))
        phase = interpolate_to_grid(freq, unwrapped, grid)
    raise RewImportError(
        f"{name}: Phase laesst sich nicht eindeutig verdichten (Schritt >= {MAX_PHASE_STEP_DEG} Grad)"
    )


def _windowed_ir(ir: dict[str, Any], name: str) -> dict[str, Any]:
    samples = np.asarray(ir["samples"], dtype=np.float64)
    dt = float(ir["sample_interval_s"])
    t0 = int(round(-float(ir["start_time_s"]) / dt))
    if not 0 <= t0 < samples.size:
        raise RewImportError(f"{name}: t=0 liegt ausserhalb der Impulsantwort")
    start = max(t0 - int(round(IR_WINDOW_BEFORE_S / dt)), 0)
    end = min(t0 + int(round(IR_WINDOW_AFTER_S / dt)), samples.size)
    return {"samples": samples[start:end].tolist(), "sample_interval_s": dt}


def build_pipeline_input(
    manifest: dict[str, Any],
    client: _RewClient,
    room_dimensions_m: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Baut das Pipeline-Eingabe-Dict (Format siehe `pipeline.py`)."""
    selected = _select_entries(manifest)
    with_phase = timing_ok(manifest)
    positions: list[dict[str, Any]] = []
    # Referenzposition zuerst, dann P02.. in fester Reihenfolge.
    order = [REFERENCE_SOURCE_POSITION] + sorted(p for p in selected if p != REFERENCE_SOURCE_POSITION)
    for source_id in order:
        is_reference = source_id == REFERENCE_SOURCE_POSITION
        position: dict[str, Any] = {"id": REFERENCE_POSITION_ID if is_reference else source_id}
        for channel, entry in sorted(selected[source_id].items()):
            name = str(entry.get("measurement_name") or f"{source_id}_{channel}")
            try:
                data = _resample_channel(client.get_frequency_response(entry["uuid"]), name, with_phase)
                if is_reference and channel in ("l", "r"):
                    data["impulse_response"] = _windowed_ir(client.get_impulse_response(entry["uuid"]), name)
            except RewImportError:
                raise
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise RewImportError(f"{name}: REW-Daten nicht verwertbar ({exc})") from exc
            notes = entry.get("notes_text")
            if notes:
                data["notes"] = str(notes)
            position[channel] = data
        positions.append(position)

    result: dict[str, Any] = {"positions": positions}
    if room_dimensions_m is not None:
        result["room_dimensions_m"] = [float(v) for v in room_dimensions_m]
    return result


def import_warnings(manifest: dict[str, Any]) -> list[str]:
    """Hinweise, die der Import nicht in die Pipeline-JSON uebernimmt:
    fehlende Timing-Freigabe (Phase verworfen) und QC-/Metadaten-Warnungen
    der verwendeten Messungen aus der guided session."""
    warnings: list[str] = []
    if not timing_ok(manifest):
        timing = manifest.get("timing_qc") or {}
        reason = timing.get("reason", "kein Timing-QC im Manifest") if isinstance(timing, dict) else "ungueltig"
        warnings.append(
            f"Phase nicht uebernommen (Timing-QC nicht OK: {reason}) - Do-No-Harm-Gate wird INCONCLUSIVE, "
            "Exporte bleiben gesperrt"
        )
    for entry in manifest.get("entries", []):
        if entry.get("status", "completed") != "completed" or entry.get("aba_state") is not None:
            continue
        name = entry.get("measurement_name")
        for finding in entry.get("qc_findings") or []:
            if isinstance(finding, dict) and finding.get("severity") != "ok":
                warnings.append(f"{name}: QC {finding.get('check')}: {finding.get('message')}")
        for finding in entry.get("metadata_findings") or []:
            if isinstance(finding, dict):
                warnings.append(f"{name}: Metadaten {finding.get('field', '')}: {finding.get('message', finding)}")
    return warnings


def import_from_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    client: _RewClient,
    room_dimensions_m: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Liest das Manifest, holt die Messungen aus REW und schreibt das JSON."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RewImportError(f"Manifest nicht lesbar: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RewImportError("Manifest-Wurzel ist kein Objekt")
    result = build_pipeline_input(manifest, client, room_dimensions_m)
    Path(output_path).write_text(json.dumps(result), encoding="utf-8")
    return result
