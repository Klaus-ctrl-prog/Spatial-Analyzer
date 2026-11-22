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
    NoHarmResult,
    NoHarmStatus,
    inconclusive_result,
    propose_broadband_correction,
    propose_mode_cuts,
    validate_no_harm,
)
from spatial_analyzer.exporters import export_kefplayer_json, export_rew_generic_eq_txt
from spatial_analyzer.grid import interpolate_to_grid, r160_grid
from spatial_analyzer.lr_sum import compare_predicted_vs_measured, predict_complex_sum
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
    # Phasenvertrag (siehe `_unwrap_phase_deg`): falls nicht None, sind die
    # Werte KONTINUIERLICH entpackt (nicht auf [-180, 180] Grad zurueck-
    # gefaltet). Fuer `complex_response.to_complex()` unschaedlich
    # (exp(1j*phase_rad) ist periodisch); ein direkter Vergleich/eine
    # direkte Anzeige von `phase_deg`-Werten muss das beruecksichtigen.
    phase_deg: np.ndarray | None  # (n_positions, n_freqs) oder None


@dataclass
class JointLRData:
    """Gemeinsames L/R-Datenpaket fuer die L+R-Summenanalyse (aktuell:
    Do-No-Harm-Mehrpunktvorschau, siehe `_build_joint_lr_data`).

    Im Unterschied zu zwei unabhaengig per `_interpolate_channel_to_common_
    grid()` aufgebauten `ChannelData`-Objekten (die je Kanal ihr eigenes
    Frequenzraster UND ihre eigene, kanalspezifische Positionsliste haben
    koennen) gilt hier zwingend: `freq_hz` ist fuer L und R IDENTISCH, und
    `position_ids` beschreibt fuer beide Kanaele dieselbe Position in
    derselben Zeile (l_spl_db[i]/l_phase_deg[i] und r_spl_db[i]/
    r_phase_deg[i] gehoeren zur selben Position `position_ids[i]`)."""

    position_ids: list[str]
    freq_hz: np.ndarray
    l_spl_db: np.ndarray
    l_phase_deg: np.ndarray
    r_spl_db: np.ndarray
    r_phase_deg: np.ndarray


def _unwrap_phase_deg(phase_deg: Any) -> np.ndarray:
    """Phasenvertrag dieses Moduls: JEDE Phase (Grad) wird VOR einer
    Interpolation auf ein Zielraster per `np.unwrap` entpackt.

    Hintergrund: `interpolate_to_grid()` interpoliert linear im log10(f)-
    Raum - ohne Entpacken wuerde ein echter Sprung wie +179 Grad ->
    (tatsaechlich weiter gedreht) +181 Grad, der als -179 Grad abgelegt ist,
    als rechnerischer Sprung von ~358 Grad interpretiert und faelschlich zu
    einem Interpolationswert nahe 0 Grad fuehren, statt nahe ±180 Grad.

    Das Ergebnis ist danach KONTINUIERLICH (nicht mehr auf [-180, 180] Grad
    zurueckgefaltet) - das ist beabsichtigt und fuer
    `complex_response.to_complex()` unschaedlich, da exp(1j*phase_rad)
    periodisch ist."""
    phase_deg = np.asarray(phase_deg, dtype=np.float64)
    return np.rad2deg(np.unwrap(np.deg2rad(phase_deg)))


def _build_id_index_map(position_ids: list[str]) -> dict[str, int]:
    """Baut eine eindeutige id->Zeilenindex-Abbildung.

    Haertung (Runde A, P0-Datenintegritaet): `position_ids.index(pid)`
    liefert bei doppelten IDs immer nur den ERSTEN Treffer und wuerde einen
    doppelten Messpunkt stillschweigend ignorieren, statt den Fehler
    sichtbar zu machen - insbesondere, wenn `run_pipeline()` ohne
    vorherige `validate_input()`-Pruefung aufgerufen wird. Diese Funktion
    baut die Zuordnung daher explizit auf und bricht fail-closed mit einer
    klaren `ValueError` ab, statt eine doppelte ID stillschweigend zu
    ueberschreiben."""
    index_map: dict[str, int] = {}
    for i, pid in enumerate(position_ids):
        if pid in index_map:
            raise ValueError(
                f"doppelte Positions-ID {pid!r} (Zeilen {index_map[pid]} und {i}) - "
                "IDs muessen je Kanal eindeutig sein (siehe validator.py)"
            )
        index_map[pid] = i
    return index_map


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
            phase_unwrapped = _unwrap_phase_deg(entry["phase_deg"])
            phase_matrix[i, :] = interpolate_to_grid(freq, phase_unwrapped, target_grid)

    return ChannelData(
        position_ids=position_ids, freq_hz=target_grid, spl_db=spl_matrix, phase_deg=phase_matrix
    )


def _build_joint_lr_data(positions: list[dict[str, Any]]) -> JointLRData:
    """Baut fuer die gemeinsame L/R-Summenanalyse (Do-No-Harm-
    Mehrpunktvorschau) ein EINZIGES gemeinsames R160-Frequenzraster und
    eine EINZIGE, fuer beide Kanaele identisch sortierte Positionsliste.

    Haertungshintergrund (Runde A, P0-Datenintegritaet): L und R werden
    fuer Mittelung/EQ-Vorschlaege bewusst UNABHAENGIG je Kanal interpoliert
    (`_interpolate_channel_to_common_grid`) - je nachdem, welche Positionen
    den jeweiligen Kanal ueberhaupt haben, koennen `ChannelData.freq_hz`
    und `ChannelData.position_ids` fuer L und R in Laenge, Reihenfolge und
    Frequenzbereich voneinander abweichen. Eine elementweise Kombination
    (z. B. `l_complex[i] + r_complex[i]`) ueber zwei solche unabhaengigen
    `ChannelData`-Objekte waere dann STILLSCHWEIGEND falsch: Zeile i in L
    muss nicht Zeile i in R entsprechen, und Frequenz j in L muss nicht
    Frequenz j in R entsprechen. Diese Funktion erzwingt stattdessen:

    1) nur Positionen mit vollstaendigem L+R-Paar (Schnittmenge der IDs),
    2) dieselbe (nach ID sortierte) Reihenfolge fuer beide Kanaele
       (`l_ids == r_ids` ist dadurch strukturell garantiert, nicht nur
       gepruefte Annahme),
    3) ein gemeinsames R160-Raster als Schnittmenge der je Kanal gueltigen
       Frequenzbereiche, auf das BEIDE Kanaele interpoliert werden.

    Loest eine `ValueError` aus (statt ein leeres/unvollstaendiges
    Ergebnis stillschweigend zurueckzugeben), wenn kein vollstaendiges
    L+R-Paar existiert, doppelte IDs unter den Paaren auftreten, nicht an
    allen Paaren eine Phase vorliegt, oder kein gemeinsamer Frequenzbereich
    existiert (`r160_grid()`) - der Aufrufer (`run_pipeline`) faengt dies
    ab und vermerkt den Grund als Hinweis statt die Pipeline abzubrechen."""
    paired = [
        p for p in positions
        if isinstance(p.get("l"), dict) and isinstance(p.get("r"), dict)
    ]
    if not paired:
        raise ValueError("keine Position mit vollstaendigem L+R-Paar gefunden")

    ids = [p["id"] for p in paired]
    if len(set(ids)) != len(ids):
        raise ValueError(
            "doppelte Positions-ID unter den vollstaendigen L+R-Paaren - "
            "gemeinsame L/R-Analyse abgebrochen (siehe validator.py)"
        )

    # Dieselbe, nach ID sortierte Reihenfolge fuer L und R statt zwei
    # unabhaengig aufgebauter Listen - garantiert l_ids == r_ids strukturell.
    paired_sorted = sorted(paired, key=lambda p: p["id"])
    sorted_ids = [p["id"] for p in paired_sorted]

    if not all("phase_deg" in p["l"] and "phase_deg" in p["r"] for p in paired_sorted):
        raise ValueError("nicht an allen vollstaendigen L+R-Positionspaaren ist eine Phase vorhanden")

    l_entries = [p["l"] for p in paired_sorted]
    r_entries = [p["r"] for p in paired_sorted]

    starts = [min(e["freq_hz"]) for e in l_entries] + [min(e["freq_hz"]) for e in r_entries]
    ends = [max(e["freq_hz"]) for e in l_entries] + [max(e["freq_hz"]) for e in r_entries]
    common_start = max(starts)
    common_end = min(ends)
    target_grid = r160_grid(common_start, common_end)

    n = len(paired_sorted)
    l_spl = np.empty((n, target_grid.size), dtype=np.float64)
    l_phase = np.empty((n, target_grid.size), dtype=np.float64)
    r_spl = np.empty((n, target_grid.size), dtype=np.float64)
    r_phase = np.empty((n, target_grid.size), dtype=np.float64)

    for i, entry in enumerate(l_entries):
        freq = np.asarray(entry["freq_hz"], dtype=np.float64)
        l_spl[i, :] = interpolate_to_grid(freq, np.asarray(entry["spl_db"], dtype=np.float64), target_grid)
        l_phase[i, :] = interpolate_to_grid(freq, _unwrap_phase_deg(entry["phase_deg"]), target_grid)
    for i, entry in enumerate(r_entries):
        freq = np.asarray(entry["freq_hz"], dtype=np.float64)
        r_spl[i, :] = interpolate_to_grid(freq, np.asarray(entry["spl_db"], dtype=np.float64), target_grid)
        r_phase[i, :] = interpolate_to_grid(freq, _unwrap_phase_deg(entry["phase_deg"]), target_grid)

    return JointLRData(
        position_ids=sorted_ids,
        freq_hz=target_grid,
        l_spl_db=l_spl,
        l_phase_deg=l_phase,
        r_spl_db=r_spl,
        r_phase_deg=r_phase,
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

        for channel_key in ("l", "r", "sub", "lr"):
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


def _lr_sum_consistency(positions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Dritte Pruefebene: reale L+R-Summenmessung an `sofa_mitte` gegen die
    phasenrichtige komplexe Summe aus L und R (lr_sum.py). Gibt None zurueck,
    wenn keine `lr`-Messung an sofa_mitte vorliegt (abwaertskompatibel);
    wirft `ValueError`, wenn sie vorliegt, aber nicht auswertbar ist (z. B.
    keine Phase an L/R, kein gemeinsamer Frequenzbereich, Band ausserhalb).
    L, R und lr werden auf EIN gemeinsames R160-Raster (Schnittmenge der
    Frequenzbereiche) interpoliert; Phase vorher entpackt."""
    ref = next((p for p in positions if p.get("id") == "sofa_mitte"), None)
    if ref is None or not isinstance(ref.get("lr"), dict):
        return None
    l_entry, r_entry, lr_entry = ref.get("l"), ref.get("r"), ref["lr"]
    if not (isinstance(l_entry, dict) and isinstance(r_entry, dict)):
        raise ValueError("sofa_mitte hat lr, aber keine vollstaendigen L/R-Daten")
    if "phase_deg" not in l_entry or "phase_deg" not in r_entry:
        raise ValueError("fuer die komplexe Vorhersage fehlt die Phase an sofa_mitte L oder R")

    entries = (l_entry, r_entry, lr_entry)
    grid = r160_grid(
        max(min(e["freq_hz"]) for e in entries), min(max(e["freq_hz"]) for e in entries)
    )

    def on_grid(entry: dict[str, Any], key: str, unwrap: bool = False) -> np.ndarray:
        values = _unwrap_phase_deg(entry[key]) if unwrap else np.asarray(entry[key], dtype=np.float64)
        return interpolate_to_grid(np.asarray(entry["freq_hz"], dtype=np.float64), values, grid)

    predicted = predict_complex_sum(
        on_grid(l_entry, "spl_db"), on_grid(l_entry, "phase_deg", True),
        on_grid(r_entry, "spl_db"), on_grid(r_entry, "phase_deg", True),
    )
    cmp = compare_predicted_vs_measured(grid, predicted, on_grid(lr_entry, "spl_db"))
    band_mask = (grid >= cmp.band_hz[0]) & (grid <= cmp.band_hz[1])
    return {
        "band_hz": list(cmp.band_hz),
        "n_points": int(np.count_nonzero(band_mask)),
        "rmse_db": cmp.rmse_db,
        "max_abs_diff_db": float(np.nanmax(np.abs(cmp.diff_db[band_mask]))),
        "meaning": "Konsistenz-Check (Vorhersage minus Messung), kein automatisches Urteil",
    }


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

    # Haertung (Runde A, P0-Datenintegritaet): die Do-No-Harm-Mehrpunkt-
    # vorschau braucht L und R auf EINEM gemeinsamen Frequenzraster und in
    # identischer Positionsreihenfolge (siehe `_build_joint_lr_data`) - die
    # vormals hier verwendeten `l_data`/`r_data` sind je Kanal unabhaengig
    # aufgebaut und duerfen dafuer NICHT direkt kombiniert werden.
    try:
        joint_lr = _build_joint_lr_data(positions)
    except ValueError as exc:
        joint_lr = None
        do_no_harm = inconclusive_result(str(exc))
        notes.append(f"Do-No-Harm-Check uebersprungen: {exc}.")

    if joint_lr is not None:
        l_complex = to_complex(joint_lr.l_spl_db, joint_lr.l_phase_deg)
        r_complex = to_complex(joint_lr.r_spl_db, joint_lr.r_phase_deg)
        do_no_harm = validate_no_harm(
            joint_lr.freq_hz, joint_lr.position_ids, l_complex, r_complex, filters_l, filters_r,
            target_db=absolute_target_db(joint_lr.freq_hz, reference_level_db, target_config),
        )
        if do_no_harm.status is NoHarmStatus.SAFE:
            notes.append(f"Do-No-Harm-Check bestanden (SAFE): {do_no_harm.reason}.")
        elif do_no_harm.status is NoHarmStatus.UNSAFE:
            notes.append(
                f"Do-No-Harm-Check: UNSAFE - {do_no_harm.reason} - Exporte sind als "
                "NICHT FREIGEGEBEN gekennzeichnet, vor Anwendung pruefen."
            )
        else:
            notes.append(
                f"Do-No-Harm-Check INCONCLUSIVE - {do_no_harm.reason} - Exporte sind als "
                "UNGEPRUEFT / NICHT FREIGEGEBEN gekennzeichnet."
            )

    lr_sum_check = None
    try:
        lr_sum_check = _lr_sum_consistency(positions)
    except ValueError as exc:
        notes.append(f"L+R-Summen-Konsistenzpruefung uebersprungen: {exc}.")

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
            #
            # Haertung (Runde A, P0-Datenintegritaet): explizite, eindeutige
            # id->Zeilenindex-Abbildungen statt `position_ids.index(pid)` -
            # ein `.index()`-Aufruf liefert bei doppelten IDs immer nur den
            # ERSTEN Treffer und wuerde das stillschweigend falsch machen.
            h_lr_rows = []
            h_sub_rows = []
            usable = True
            try:
                l_index_map = _build_id_index_map(l_data.position_ids)
                r_index_map = _build_id_index_map(r_data.position_ids)
                sub_index_map = _build_id_index_map(sub_data.position_ids)
            except ValueError as exc:
                notes.append(f"Subwoofer-Optimierung uebersprungen: {exc}.")
                l_index_map = r_index_map = sub_index_map = None
                usable = False

            for pid in sub_data.position_ids if usable else []:
                if pid not in l_index_map or pid not in r_index_map:
                    notes.append(
                        f"Position '{pid}' hat Sub-, aber keine vollstaendigen L/R-Daten - "
                        "Subwoofer-Optimierung uebersprungen."
                    )
                    usable = False
                    break
                li = l_index_map[pid]
                ri = r_index_map[pid]
                l_spl_i = interpolate_to_grid(l_data.freq_hz, l_data.spl_db[li, :], sub_data.freq_hz)
                l_phase_i = interpolate_to_grid(l_data.freq_hz, l_data.phase_deg[li, :], sub_data.freq_hz)
                r_spl_i = interpolate_to_grid(r_data.freq_hz, r_data.spl_db[ri, :], sub_data.freq_hz)
                r_phase_i = interpolate_to_grid(r_data.freq_hz, r_data.phase_deg[ri, :], sub_data.freq_hz)
                h_lr_rows.append(to_complex(l_spl_i, l_phase_i) + to_complex(r_spl_i, r_phase_i))
                si = sub_index_map[pid]
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
        do_no_harm=do_no_harm,
        lr_sum_check=lr_sum_check,
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
    # Nur SAFE ist freigegeben; UNSAFE/INCONCLUSIVE werden in Profilname,
    # REW-Kommentar und Begleitdatei deutlich als NICHT FREIGEGEBEN markiert.
    if do_no_harm.is_released:
        profile_name, rew_header = "spatial_analyzer", None
    else:
        label = (
            "UNSAFE - NICHT FREIGEGEBEN"
            if do_no_harm.status is NoHarmStatus.UNSAFE
            else "UNGEPRUEFT - NICHT FREIGEGEBEN"
        )
        profile_name = f"spatial_analyzer [{label}]"
        rew_header = f"{label}\nDo-No-Harm-Status: {do_no_harm.status.value}\n{do_no_harm.reason}"
        (output_dir / "NICHT_FREIGEGEBEN.txt").write_text(
            f"{label}\n\nDo-No-Harm-Status: {do_no_harm.status.value}\nGrund: {do_no_harm.reason}\n\n"
            "Die Exportdateien kefplayer_profile.json, rew_eq_L.txt und rew_eq_R.txt in diesem "
            "Ordner duerfen nicht ungeprueft angewendet werden.\n",
            encoding="utf-8",
        )
    export_kefplayer_json(output_dir / "kefplayer_profile.json", filters_l, filters_r, name=profile_name)
    export_rew_generic_eq_txt(output_dir / "rew_eq_L.txt", filters_l, header_comment=rew_header)
    export_rew_generic_eq_txt(output_dir / "rew_eq_R.txt", filters_r, header_comment=rew_header)

    return report
