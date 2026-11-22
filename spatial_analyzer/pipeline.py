"""End-to-End-Pipeline: von einer JSON-Eingabedatei bis zu Report + Exporten.

Eingabeformat (siehe `docs/SPECIFICATION.md` Abschnitt 4, G.4): da der
echte REW-`.mdat`-Byte-Parser in dieser Recherchephase nicht verifiziert ist
(siehe mdat.py), arbeitet die CLI/Pipeline auf einer dokumentierten JSON-
Zwischenform, nicht auf echten Binaer-`.mdat`-Dateien:

{
  "room_dimensions_m": [laenge, breite, hoehe],
  "positions": [
    {
      "id": "sofa_mitte",
      "notes": "..." (optional, REW-Notes-Text fuer die ganze Position),
      "l": {"freq_hz": [...], "spl_db": [...], "phase_deg": [...] (optional),
            "impulse_response": {"samples": [...], "sample_interval_s": ...} (optional),
            "notes": "..." (optional, REW-Notes-Text fuer diesen Kanal)},
      "r": {...gleiche Struktur...},
      "sub": {...optional, gleiche Struktur wie l/r...}
    },
    ...
  ]
}

Mindestens eine Position mit `id` gleich "sofa_mitte" sollte vorhanden sein
(Haupthoerplatz); andernfalls wird die erste Position als Referenz fuer die
Abklinganalyse verwendet und das wird in den Berichts-Hinweisen vermerkt.

Das optionale `notes`-Feld (pro Position und/oder pro Kanal `l`/`r`/`sub`)
ist der vom Nutzer in REW hinterlegte Klartext, typischerweise die DSP-
Einstellungen von Geraeten, die REW nicht automatisch auslesen kann (siehe
docs/REW_NOTES_TEMPLATE.md). Es landet unveraendert in `Report.measurement_
notes` bzw. im Berichtsabschnitt "Mess- & DSP-Notizen" - strikt getrennt
von `Report.notes` (den intern von der Pipeline erzeugten System-/Warn-
hinweisen). Fehlt `notes`, bleibt die jeweilige Position/der Kanal im
Berichtsabschnitt unerwaehnt; bestehende Eingabedateien ohne dieses Feld
funktionieren unveraendert weiter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from spatial_analyzer.averaging import average_variant_b, energetic_mean_db, spread_metrics
from spatial_analyzer.bass_resonance import (
    compare_to_room_model,
    evaluate_decay_against_threshold,
    find_peak_candidates,
    narrowband_decay_time,
    spatial_consistency,
)
from spatial_analyzer.biquad_preview import PeakingFilter
from spatial_analyzer.complex_response import to_complex
from spatial_analyzer.etc import classify_bass_dip
from spatial_analyzer.eq_solver import (
    EqProposal,
    ModeConfirmation,
    propose_broadband_correction,
    propose_mode_cuts,
    validate_no_harm,
)
from spatial_analyzer.exporters import export_kefplayer_json, export_rew_generic_eq_txt
from spatial_analyzer.grid import interpolate_to_grid, r160_grid
from spatial_analyzer.models import RoomModel
from spatial_analyzer.reporting import MeasurementNote, Report, write_frequency_table_csv
from spatial_analyzer.sub_optimizer import optimize as optimize_sub
from spatial_analyzer.target_curve import TargetCurveConfig, absolute_target_db

MODE_BAND_HZ = (20.0, 200.0)
REFERENCE_BAND_HZ = (400.0, 2000.0)


@dataclass
class ChannelData:
    position_ids: list[str]
    freq_hz: np.ndarray  # gemeinsames R160-Raster
    spl_db: np.ndarray  # (n_positions, n_freqs)
    phase_deg: np.ndarray | None  # (n_positions, n_freqs) oder None


def _interpolate_channel_to_common_grid(
    positions: list[dict[str, Any]], channel_key: str
) -> ChannelData:
    position_ids = [p["id"] for p in positions if channel_key in p]
    raw = [p[channel_key] for p in positions if channel_key in p]
    if not raw:
        raise ValueError(f"Keine Positionen mit Kanal '{channel_key}' gefunden")

    starts = [min(entry["freq_hz"]) for entry in raw]
    ends = [max(entry["freq_hz"]) for entry in raw]
    common_start = max(starts)
    common_end = min(ends)
    target_grid = r160_grid(common_start, common_end)

    spl_matrix = np.empty((len(raw), target_grid.size), dtype=np.float64)
    has_phase = all("phase_deg" in entry for entry in raw)
    phase_matrix = np.empty((len(raw), target_grid.size), dtype=np.float64) if has_phase else None

    for i, entry in enumerate(raw):
        freq = np.asarray(entry["freq_hz"], dtype=np.float64)
        spl = np.asarray(entry["spl_db"], dtype=np.float64)
        spl_matrix[i, :] = interpolate_to_grid(freq, spl, target_grid)
        if has_phase:
            phase = np.asarray(entry["phase_deg"], dtype=np.float64)
            phase_matrix[i, :] = interpolate_to_grid(freq, phase, target_grid)

    return ChannelData(
        position_ids=position_ids, freq_hz=target_grid, spl_db=spl_matrix, phase_deg=phase_matrix
    )


def _confirm_mode_candidates(
    reference_freq_hz: np.ndarray,
    averaged_spl_db: np.ndarray,
    position_spl_db: np.ndarray,
    room: RoomModel | None,
    reference_ir: dict[str, Any] | None,
) -> list[ModeConfirmation]:
    candidates = find_peak_candidates(reference_freq_hz, averaged_spl_db, band_hz=MODE_BAND_HZ)
    if not candidates:
        return []

    per_position_candidates = [
        find_peak_candidates(reference_freq_hz, position_spl_db[i, :], band_hz=MODE_BAND_HZ)
        for i in range(position_spl_db.shape[0])
    ]
    per_position_freqs = [[c.freq_hz for c in cands] for cands in per_position_candidates]

    theoretical = list(room.axial_mode_candidates(max_order=2).values()) if room else []

    confirmations: list[ModeConfirmation] = []
    for cand in candidates:
        classification = classify_bass_dip(cand.freq_hz, sum(per_position_freqs, []), theoretical)
        # Hinweis: sum(per_position_freqs, []) nutzt absichtlich ALLE
        # Positions-Frequenzen (nicht nur die naechsten je Position), da
        # classify_bass_dip selbst die Toleranzpruefung je Eintrag macht.

        decay_confirmed: bool | None = None
        if reference_ir is not None:
            ir_samples = np.asarray(reference_ir["samples"], dtype=np.float64)
            dt = float(reference_ir["sample_interval_s"])
            try:
                decay_s = narrowband_decay_time(ir_samples, dt, cand.freq_hz)
                decay_eval = evaluate_decay_against_threshold(decay_s, cand.freq_hz)
                decay_confirmed = decay_eval.exceeds_perceptual_threshold
            except ValueError:
                decay_confirmed = None

        confirmations.append(
            ModeConfirmation(
                freq_hz=cand.freq_hz,
                classification=classification,
                decay_confirmed=decay_confirmed,
                q_factor=cand.q_factor,
            )
        )
    return confirmations


def _collect_measurement_notes(positions: list[dict[str, Any]]) -> list[MeasurementNote]:
    """Liest das optionale `notes`-Feld pro Position und pro Kanal
    (`l`/`r`/`sub`) aus der Eingabe-JSON ein - der vom Nutzer in REW
    hinterlegte Klartext (z. B. DSP-Einstellungen, siehe
    docs/REW_NOTES_TEMPLATE.md). Strikt getrennt von den intern von der
    Pipeline erzeugten System-/Warnhinweisen (siehe `notes`-Liste in
    `run_pipeline`).

    Abwaertskompatibel: fehlt `notes` (Position oder Kanal), wird dafuer
    kein Eintrag erzeugt - bestehende Eingabedateien ohne dieses Feld
    funktionieren unveraendert weiter.
    """
    collected: list[MeasurementNote] = []
    for position in positions:
        position_id = str(position.get("id", "?"))

        position_note = position.get("notes")
        if isinstance(position_note, str) and position_note.strip():
            collected.append(
                MeasurementNote(position_id=position_id, channel=None, text=position_note.strip())
            )

        for channel_key in ("l", "r", "sub"):
            channel = position.get(channel_key)
            if not isinstance(channel, dict):
                continue
            channel_note = channel.get("notes")
            if isinstance(channel_note, str) and channel_note.strip():
                collected.append(
                    MeasurementNote(
                        position_id=position_id, channel=channel_key, text=channel_note.strip()
                    )
                )
    return collected


def run_pipeline(input_path: str | Path, output_dir: str | Path) -> Report:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    positions = data["positions"]
    if not positions:
        raise ValueError("Eingabedatei enthaelt keine Positionen")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    room = None
    room_dims = data.get("room_dimensions_m")
    if room_dims:
        room = RoomModel(length_m=room_dims[0], width_m=room_dims[1], height_m=room_dims[2])

    notes: list[str] = []
    measurement_notes = _collect_measurement_notes(positions)

    l_data = _interpolate_channel_to_common_grid(positions, "l")
    r_data = _interpolate_channel_to_common_grid(positions, "r")

    averaged_l = average_variant_b(l_data.spl_db, l_data.freq_hz, align_band_hz=REFERENCE_BAND_HZ)
    averaged_r = average_variant_b(r_data.spl_db, r_data.freq_hz, align_band_hz=REFERENCE_BAND_HZ)
    spread_l = spread_metrics(l_data.spl_db)
    spread_r = spread_metrics(r_data.spl_db)

    ref_band_mask_l = (l_data.freq_hz >= REFERENCE_BAND_HZ[0]) & (l_data.freq_hz <= REFERENCE_BAND_HZ[1])
    reference_level_db = float(energetic_mean_db(averaged_l[np.newaxis, ref_band_mask_l], axis=1)[0])
    target_config = TargetCurveConfig()
    target_l = absolute_target_db(l_data.freq_hz, reference_level_db, target_config)
    target_r = absolute_target_db(r_data.freq_hz, reference_level_db, target_config)

    reference_position = next(
        (p for p in positions if p.get("id") == "sofa_mitte"), positions[0]
    )
    if reference_position.get("id") != "sofa_mitte":
        notes.append(
            f"Keine Position mit id='sofa_mitte' gefunden - '{reference_position.get('id')}' "
            "als Referenz fuer die Abklinganalyse verwendet."
        )
    reference_ir = reference_position.get("l", {}).get("impulse_response")
    if reference_ir is None:
        notes.append(
            "Keine Impulsantwort an der Referenzposition vorhanden - Abklinganalyse "
            "wird uebersprungen, Modenkandidaten bleiben decay_confirmed=None und werden "
            "deshalb NICHT fuer Cut-Vorschlaege verwendet (siehe eq_solver.py)."
        )

    confirmations = _confirm_mode_candidates(
        l_data.freq_hz, averaged_l, l_data.spl_db, room, reference_ir
    )
    comb_filter_freqs = [c.freq_hz for c in confirmations if c.classification == "comb_filter_candidate"]

    eq_proposals_l = propose_mode_cuts(l_data.freq_hz, averaged_l, target_l, confirmations)
    eq_proposals_l += propose_broadband_correction(
        l_data.freq_hz, averaged_l, target_l, excluded_freqs_hz=comb_filter_freqs
    )
    eq_proposals_r = propose_mode_cuts(r_data.freq_hz, averaged_r, target_r, confirmations)
    eq_proposals_r += propose_broadband_correction(
        r_data.freq_hz, averaged_r, target_r, excluded_freqs_hz=comb_filter_freqs
    )

    filters_l = [p.filter for p in eq_proposals_l]
    filters_r = [p.filter for p in eq_proposals_r]

    if l_data.phase_deg is not None and r_data.phase_deg is not None:
        l_complex = to_complex(l_data.spl_db, l_data.phase_deg)
        r_complex = to_complex(r_data.spl_db, r_data.phase_deg)
        is_safe, spread_before, spread_after = validate_no_harm(
            l_data.freq_hz, l_data.position_ids, l_complex, r_complex, filters_l, filters_r
        )
        if not is_safe:
            notes.append(
                f"Do-No-Harm-Check: vorgeschlagene Filter erhoehen die raeumliche Streuung "
                f"({spread_before:.2f} dB -> {spread_after:.2f} dB) - vor Anwendung pruefen."
            )
        else:
            notes.append(
                f"Do-No-Harm-Check bestanden: raeumliche Streuung {spread_before:.2f} dB -> "
                f"{spread_after:.2f} dB."
            )
    else:
        notes.append("Keine Phase fuer alle Positionen vorhanden - Do-No-Harm-Check uebersprungen.")

    sub_result = None
    has_sub_everywhere = all(p.get("sub") for p in positions)
    if has_sub_everywhere and l_data.phase_deg is not None and r_data.phase_deg is not None:
        try:
            sub_data = _interpolate_channel_to_common_grid(positions, "sub")
        except ValueError as exc:
            notes.append(f"Subwoofer-Optimierung uebersprungen: {exc}")
            sub_data = None

        if sub_data is not None and sub_data.phase_deg is not None:
            # Die Sub-Messungen haben i.A. ein anderes Frequenzraster als
            # L/R (siehe grid.py) - L+R wird deshalb pro Position auf das
            # Sub-Raster interpoliert, nicht umgekehrt, da der Uebergangs-
            # bereich (50-120 Hz, siehe sub_optimizer.py) massgeblich ist.
            h_lr_rows = []
            h_sub_rows = []
            usable = True
            for pid in sub_data.position_ids:
                if pid not in l_data.position_ids or pid not in r_data.position_ids:
                    notes.append(
                        f"Position '{pid}' hat Sub-, aber keine vollstaendigen L/R-Daten - "
                        "Subwoofer-Optimierung uebersprungen."
                    )
                    usable = False
                    break
                li = l_data.position_ids.index(pid)
                ri = r_data.position_ids.index(pid)
                l_spl_i = interpolate_to_grid(l_data.freq_hz, l_data.spl_db[li, :], sub_data.freq_hz)
                l_phase_i = interpolate_to_grid(l_data.freq_hz, l_data.phase_deg[li, :], sub_data.freq_hz)
                r_spl_i = interpolate_to_grid(r_data.freq_hz, r_data.spl_db[ri, :], sub_data.freq_hz)
                r_phase_i = interpolate_to_grid(r_data.freq_hz, r_data.phase_deg[ri, :], sub_data.freq_hz)
                h_lr_rows.append(to_complex(l_spl_i, l_phase_i) + to_complex(r_spl_i, r_phase_i))
                si = sub_data.position_ids.index(pid)
                h_sub_rows.append(to_complex(sub_data.spl_db[si, :], sub_data.phase_deg[si, :]))

            if usable:
                h_lr = np.vstack(h_lr_rows)
                h_sub = np.vstack(h_sub_rows)
                target_sum = absolute_target_db(sub_data.freq_hz, reference_level_db, target_config)
                try:
                    sub_result = optimize_sub(sub_data.freq_hz, h_lr, h_sub, target_sum)
                except ValueError as exc:
                    notes.append(f"Subwoofer-Optimierung uebersprungen: {exc}")
    else:
        notes.append("Keine vollstaendigen Sub-Messungen fuer alle Positionen - Optimierung uebersprungen.")

    report = Report(
        generated_at=datetime.now(timezone.utc).isoformat(),
        room_dimensions_m=tuple(room_dims) if room_dims else None,
        measurement_summary={
            "l_reference_level_db_400_2000hz": reference_level_db,
            "l_averaged_spl_db_at_1khz": float(
                np.interp(1000.0, l_data.freq_hz, averaged_l)
            ),
            "n_positions_l": len(l_data.position_ids),
            "n_positions_r": len(r_data.position_ids),
            "l_spread_db_mean": float(np.nanmean(spread_l.std_db)),
            "r_spread_db_mean": float(np.nanmean(spread_r.std_db)),
        },
        mode_confirmations=confirmations,
        sub_optimizer_result=sub_result,
        eq_proposals_l=eq_proposals_l,
        eq_proposals_r=eq_proposals_r,
        measurement_notes=measurement_notes,
        notes=notes,
    )

    report.write_json(output_dir / "report.json")
    report.write_html(output_dir / "report.html")
    write_frequency_table_csv(
        output_dir / "frequency_table_l.csv",
        l_data.freq_hz,
        {"averaged_spl_db": averaged_l, "target_db": target_l, "spread_std_db": spread_l.std_db},
    )
    write_frequency_table_csv(
        output_dir / "frequency_table_r.csv",
        r_data.freq_hz,
        {"averaged_spl_db": averaged_r, "target_db": target_r, "spread_std_db": spread_r.std_db},
    )
    export_kefplayer_json(output_dir / "kefplayer_profile.json", filters_l, filters_r)
    export_rew_generic_eq_txt(output_dir / "rew_eq_L.txt", filters_l)
    export_rew_generic_eq_txt(output_dir / "rew_eq_R.txt", filters_r)

    return report
