"""A/B-Vergleich zweier abgeschlossener `run_pipeline()`-Durchlaeufe
(vorher/nachher), passend zur eigenen Messregel "immer nur eine Variable
aendern, danach neu messen" (siehe Projekt-Anweisung, Abschnitt MESSREGELN).

Liest aus beiden Ausgabeordnern:
- `report.json` (measurement_summary, mode_confirmations, eq_proposals)
- `frequency_table_l.csv` / `frequency_table_r.csv` (freq_hz, averaged_spl_db)

Pegelangleichung: die "nachher"-Kurve wird so verschoben, dass ihr
Referenzpegel (energetischer Mittelwert 400 Hz - 2 kHz) exakt dem
"vorher"-Referenzpegel entspricht (analog REW "Align SPL", siehe eigene
Messregel 4: "Referenzpegel fuer alle Vergleiche: Mittelwert 400 Hz-2 kHz.
Overlays erst nach Angleichen auf diesen Pegel bewerten."). Es wird NICHT
auf 0 dB normiert, sondern die "nachher"-Kurve an die "vorher"-Kurve
angeglichen, damit reale Pegelunterschiede ausserhalb des Referenzbands
(z. B. im Bass) sichtbar bleiben, aber ein unterschiedlicher Ausgabepegel
zwischen den beiden Mess-Sessions die Delta-Kurve nicht verfaelscht.

Die vollen Delta-Kurven werden als CSV geschrieben (wie die uebrigen
Frequenztabellen in reporting.py), der JSON/HTML-Bericht enthaelt bewusst
nur verdichtete Kennwerte (Mittel/Max/Min, je Gesamtband und Modenband) -
siehe docs/SPECIFICATION.md Abschnitt 3.F zur Trennung CSV (frequenz-
indizierte Rohdaten) vs. JSON/HTML (nachvollziehbare Zusammenfassung).
"""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from spatial_analyzer.averaging import DEFAULT_ALIGN_BAND_HZ, energetic_mean_db
from spatial_analyzer.grid import interpolate_to_grid, r160_grid
from spatial_analyzer.reporting import write_frequency_table_csv

MODE_BAND_HZ = (20.0, 200.0)
MODE_MATCH_TOLERANCE_HZ = 3.0


@dataclass
class CurveDelta:
    freq_hz: np.ndarray
    delta_db: np.ndarray  # (nachher, angeglichen) - vorher
    alignment_shift_db: float  # auf die "nachher"-Kurve angewandte Verschiebung


@dataclass
class ModeComparison:
    freq_hz: float
    classification_before: str | None
    classification_after: str | None
    decay_confirmed_before: bool | None
    decay_confirmed_after: bool | None


@dataclass
class CompareReport:
    generated_at: str
    before_dir: str
    after_dir: str
    delta_l: CurveDelta | None
    delta_r: CurveDelta | None
    spread_before_l_db: float | None
    spread_after_l_db: float | None
    spread_before_r_db: float | None
    spread_after_r_db: float | None
    mode_comparisons: list[ModeComparison] = field(default_factory=list)
    eq_proposal_count_before_l: int = 0
    eq_proposal_count_after_l: int = 0
    eq_proposal_count_before_r: int = 0
    eq_proposal_count_after_r: int = 0
    notes: list[str] = field(default_factory=list)
    align_band_hz: tuple[float, float] | None = DEFAULT_ALIGN_BAND_HZ

    def _delta_summary(self, delta: CurveDelta | None) -> dict[str, Any] | None:
        if delta is None:
            return None
        band_mask = (delta.freq_hz >= MODE_BAND_HZ[0]) & (delta.freq_hz <= MODE_BAND_HZ[1])
        return {
            "alignment_shift_db": delta.alignment_shift_db,
            "mean_delta_db": float(np.nanmean(delta.delta_db)),
            "max_delta_db": float(np.nanmax(delta.delta_db)),
            "min_delta_db": float(np.nanmin(delta.delta_db)),
            "mean_delta_db_mode_band_20_200hz": (
                float(np.nanmean(delta.delta_db[band_mask])) if band_mask.any() else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "vorher_verzeichnis": self.before_dir,
            "nachher_verzeichnis": self.after_dir,
            "messwerte": {
                "spread_db_mean_l": {"vorher": self.spread_before_l_db, "nachher": self.spread_after_l_db},
                "spread_db_mean_r": {"vorher": self.spread_before_r_db, "nachher": self.spread_after_r_db},
            },
            "align_band_hz": list(self.align_band_hz) if self.align_band_hz is not None else None,
            "delta_kurven": {
                "l": self._delta_summary(self.delta_l),
                "r": self._delta_summary(self.delta_r),
                "hinweis": "volle frequenzaufgeloeste Delta-Kurven in compare_delta_l.csv / "
                "compare_delta_r.csv, nicht in diesem JSON",
            },
            "modellannahmen": {
                "mode_comparisons": [
                    {
                        "freq_hz": m.freq_hz,
                        "classification_before": m.classification_before,
                        "classification_after": m.classification_after,
                        "decay_confirmed_before": m.decay_confirmed_before,
                        "decay_confirmed_after": m.decay_confirmed_after,
                    }
                    for m in self.mode_comparisons
                ],
            },
            "empfehlungen": {
                "eq_proposal_count_l": {
                    "vorher": self.eq_proposal_count_before_l, "nachher": self.eq_proposal_count_after_l
                },
                "eq_proposal_count_r": {
                    "vorher": self.eq_proposal_count_before_r, "nachher": self.eq_proposal_count_after_r
                },
            },
            "notes": self.notes,
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def write_html(self, path: str | Path) -> None:
        Path(path).write_text(render_compare_html(self), encoding="utf-8")


def _read_frequency_table(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        columns: dict[str, list[float]] = {name: [] for name in header}
        for row in reader:
            for name, value in zip(header, row):
                columns[name].append(float("nan") if value == "" else float(value))
    return {name: np.array(values, dtype=np.float64) for name, values in columns.items()}


def _load_report(output_dir: Path) -> dict[str, Any]:
    report_path = output_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"report.json nicht gefunden in {output_dir}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _compute_curve_delta(
    before_table: dict[str, np.ndarray], after_table: dict[str, np.ndarray],
    align_band_hz: tuple[float, float] | None = DEFAULT_ALIGN_BAND_HZ,
) -> CurveDelta:
    before_freq = before_table["freq_hz"]
    after_freq = after_table["freq_hz"]
    common_start = max(before_freq.min(), after_freq.min())
    common_end = min(before_freq.max(), after_freq.max())
    target_grid = r160_grid(common_start, common_end)

    before_on_grid = interpolate_to_grid(before_freq, before_table["averaged_spl_db"], target_grid)
    after_on_grid = interpolate_to_grid(after_freq, after_table["averaged_spl_db"], target_grid)

    if align_band_hz is None:
        shift = 0.0
    else:
        if len(align_band_hz) != 2 or align_band_hz[0] >= align_band_hz[1]:
            raise ValueError(f"Ungueltiges Referenzband {align_band_hz!r} Hz")
        band_mask = (target_grid >= align_band_hz[0]) & (target_grid <= align_band_hz[1])
        if not band_mask.any():
            raise ValueError(f"Referenzband {align_band_hz} Hz liegt ausserhalb des gemeinsamen Frequenzbereichs")
        ref_before = float(energetic_mean_db(before_on_grid[band_mask]))
        ref_after = float(energetic_mean_db(after_on_grid[band_mask]))
        shift = ref_before - ref_after

    delta_db = (after_on_grid + shift) - before_on_grid
    return CurveDelta(freq_hz=target_grid, delta_db=delta_db, alignment_shift_db=shift)


def _match_mode_comparisons(
    before_modes: list[dict[str, Any]], after_modes: list[dict[str, Any]]
) -> list[ModeComparison]:
    comparisons: list[ModeComparison] = []
    used_after = set()

    for b in before_modes:
        match_idx = None
        for i, a in enumerate(after_modes):
            if i in used_after:
                continue
            if abs(a["freq_hz"] - b["freq_hz"]) <= MODE_MATCH_TOLERANCE_HZ:
                match_idx = i
                break
        if match_idx is not None:
            used_after.add(match_idx)
            a = after_modes[match_idx]
            comparisons.append(
                ModeComparison(
                    freq_hz=b["freq_hz"],
                    classification_before=b["classification"],
                    classification_after=a["classification"],
                    decay_confirmed_before=b["decay_confirmed"],
                    decay_confirmed_after=a["decay_confirmed"],
                )
            )
        else:
            comparisons.append(
                ModeComparison(
                    freq_hz=b["freq_hz"],
                    classification_before=b["classification"],
                    classification_after=None,
                    decay_confirmed_before=b["decay_confirmed"],
                    decay_confirmed_after=None,
                )
            )

    for i, a in enumerate(after_modes):
        if i not in used_after:
            comparisons.append(
                ModeComparison(
                    freq_hz=a["freq_hz"],
                    classification_before=None,
                    classification_after=a["classification"],
                    decay_confirmed_before=None,
                    decay_confirmed_after=a["decay_confirmed"],
                )
            )

    return comparisons


def run_compare(before_dir: str | Path, after_dir: str | Path, output_dir: str | Path, align_band_hz: tuple[float, float] | None = DEFAULT_ALIGN_BAND_HZ) -> CompareReport:
    before_dir = Path(before_dir)
    after_dir = Path(after_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    before_report = _load_report(before_dir)
    after_report = _load_report(after_dir)

    notes: list[str] = []

    deltas: dict[str, CurveDelta | None] = {"l": None, "r": None}
    for channel in ("l", "r"):
        before_csv = before_dir / f"frequency_table_{channel}.csv"
        after_csv = after_dir / f"frequency_table_{channel}.csv"
        if not before_csv.exists() or not after_csv.exists():
            notes.append(
                f"Kanal '{channel}': frequency_table_{channel}.csv fehlt in mindestens einem der "
                "beiden Verzeichnisse - Delta-Kurve wird uebersprungen."
            )
            continue
        try:
            deltas[channel] = _compute_curve_delta(
                _read_frequency_table(before_csv), _read_frequency_table(after_csv), align_band_hz=align_band_hz
            )
        except ValueError as exc:
            notes.append(f"Kanal '{channel}': Delta-Kurve uebersprungen ({exc}).")

    for channel in ("l", "r"):
        delta = deltas[channel]
        if delta is not None:
            write_frequency_table_csv(
                output_dir / f"compare_delta_{channel}.csv", delta.freq_hz, {"delta_db": delta.delta_db}
            )

    before_summary = before_report.get("messwerte", {})
    after_summary = after_report.get("messwerte", {})

    before_modes = before_report.get("modellannahmen", {}).get("mode_confirmations", [])
    after_modes = after_report.get("modellannahmen", {}).get("mode_confirmations", [])
    mode_comparisons = _match_mode_comparisons(before_modes, after_modes)

    before_eq_l = before_report.get("empfehlungen", {}).get("eq_proposals_l", [])
    after_eq_l = after_report.get("empfehlungen", {}).get("eq_proposals_l", [])
    before_eq_r = before_report.get("empfehlungen", {}).get("eq_proposals_r", [])
    after_eq_r = after_report.get("empfehlungen", {}).get("eq_proposals_r", [])

    sub_before = before_report.get("modellannahmen", {}).get("sub_optimizer")
    sub_after = after_report.get("modellannahmen", {}).get("sub_optimizer")
    if sub_before and sub_after:
        notes.append(
            f"Subwoofer-Optimierung: Gain {sub_before['best_gain_db']:+.1f} dB -> "
            f"{sub_after['best_gain_db']:+.1f} dB, Delay {sub_before['best_delay_ms']:.1f} ms -> "
            f"{sub_after['best_delay_ms']:.1f} ms, Polaritaet {sub_before['best_polarity_deg']:.0f} Grad "
            f"-> {sub_after['best_polarity_deg']:.0f} Grad."
        )
    elif sub_before or sub_after:
        notes.append(
            "Subwoofer-Optimierungsergebnis ist nur in einem der beiden Durchlaeufe vorhanden - "
            "kein direkter Vergleich moeglich."
        )

    report = CompareReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        before_dir=str(before_dir),
        after_dir=str(after_dir),
        delta_l=deltas["l"],
        delta_r=deltas["r"],
        spread_before_l_db=before_summary.get("l_spread_db_mean"),
        spread_after_l_db=after_summary.get("l_spread_db_mean"),
        spread_before_r_db=before_summary.get("r_spread_db_mean"),
        spread_after_r_db=after_summary.get("r_spread_db_mean"),
        mode_comparisons=mode_comparisons,
        eq_proposal_count_before_l=len(before_eq_l),
        eq_proposal_count_after_l=len(after_eq_l),
        eq_proposal_count_before_r=len(before_eq_r),
        eq_proposal_count_after_r=len(after_eq_r),
        notes=notes,
        align_band_hz=align_band_hz,
    )

    report.write_json(output_dir / "compare_report.json")
    report.write_html(output_dir / "compare_report.html")

    return report


def render_compare_html(report: CompareReport) -> str:
    e = html.escape

    def delta_block(label: str, delta: CurveDelta | None) -> str:
        if delta is None:
            return f"<p><em>Kanal {label}: keine Delta-Kurve (siehe Hinweise).</em></p>"
        band_mask = (delta.freq_hz >= MODE_BAND_HZ[0]) & (delta.freq_hz <= MODE_BAND_HZ[1])
        mode_band_mean = float(np.nanmean(delta.delta_db[band_mask])) if band_mask.any() else float("nan")
        return (
            f"<p>Kanal {label}: Angleich-Verschiebung der Nachher-Kurve: "
            f"{delta.alignment_shift_db:+.2f} dB<br>"
            f"Delta (Nachher - Vorher, nach Angleich): Mittel {np.nanmean(delta.delta_db):+.2f} dB, "
            f"Min {np.nanmin(delta.delta_db):+.2f} dB, Max {np.nanmax(delta.delta_db):+.2f} dB<br>"
            f"Mittel im Modenband 20-200 Hz: {mode_band_mean:+.2f} dB<br>"
            f"Volle Kurve: <code>compare_delta_{label.lower()}.csv</code></p>"
        )

    def mode_rows() -> str:
        rows = []
        for m in report.mode_comparisons:
            cb = m.classification_before or "-"
            ca = m.classification_after or "-"
            rows.append(
                f"<tr><td>{m.freq_hz:.1f} Hz</td><td>{e(cb)}</td><td>{e(ca)}</td></tr>"
            )
        return "\n".join(rows) or "<tr><td colspan='3'><em>keine</em></td></tr>"

    def spread_row(label: str, before: float | None, after: float | None) -> str:
        b = "unbekannt" if before is None else f"{before:.2f} dB"
        a = "unbekannt" if after is None else f"{after:.2f} dB"
        return f"<tr><td>{e(label)}</td><td>{b}</td><td>{a}</td></tr>"

    notes_html = "".join(f"<li>{e(n)}</li>" for n in report.notes) or "<li><em>keine</em></li>"

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Spatial Analyzer A/B-Vergleich - {e(report.generated_at)}</title>
<style>
body {{ font-family: sans-serif; margin: 2em; color: #222; }}
h1, h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5em; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
.model-banner {{ background: #fff3cd; padding: 0.5em; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Spatial Analyzer A/B-Vergleich</h1>
<p>Erstellt: {e(report.generated_at)}</p>
<p>Vorher: <code>{e(report.before_dir)}</code><br>Nachher: <code>{e(report.after_dir)}</code></p>

<h2>Raeumliche Streuung (Mittelwert ueber alle Positionen)</h2>
<table>
<tr><th>Kanal</th><th>Vorher</th><th>Nachher</th></tr>
{spread_row("L", report.spread_before_l_db, report.spread_after_l_db)}
{spread_row("R", report.spread_before_r_db, report.spread_after_r_db)}
</table>

<h2>Delta-Kurven (Referenzband: {e(str(report.align_band_hz)) if report.align_band_hz is not None else "keine Angleichung (0 dB Verschiebung)"})</h2>
<div class="model-banner">Die Nachher-Kurve wurde auf den Referenzpegel der
Vorher-Kurve verschoben (analog REW "Align SPL") - Unterschiede im
Referenzband selbst sind dadurch per Definition nahe 0 dB.</div>
{delta_block("L", report.delta_l)}
{delta_block("R", report.delta_r)}

<h2>Modellannahmen: Modenkandidaten vorher/nachher</h2>
<table>
<tr><th>Frequenz</th><th>Klassifikation vorher</th><th>Klassifikation nachher</th></tr>
{mode_rows()}
</table>

<h2>Empfehlungen: Anzahl EQ-Vorschlaege</h2>
<table>
<tr><th>Kanal</th><th>Vorher</th><th>Nachher</th></tr>
<tr><td>L</td><td>{report.eq_proposal_count_before_l}</td><td>{report.eq_proposal_count_after_l}</td></tr>
<tr><td>R</td><td>{report.eq_proposal_count_before_r}</td><td>{report.eq_proposal_count_after_r}</td></tr>
</table>

<h2>Hinweise</h2>
<ul>
{notes_html}
</ul>
</body>
</html>
"""
