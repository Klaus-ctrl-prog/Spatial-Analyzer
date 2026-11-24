"""Nachvollziehbare Berichte (JSON/CSV/HTML) mit klarer Trennung von
Messwerten, Modellannahmen und Empfehlungen.

Siehe docs/SPECIFICATION.md Abschnitt 3.F: "Berichte: Nachvollziehbare
CSV-, JSON- und HTML-Berichte mit klarer Trennung von Messwerten,
Modellannahmen und Filterempfehlungen."
"""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spatial_analyzer.eq_solver import EqProposal, ModeConfirmation, NoHarmResult, NoHarmStatus
from spatial_analyzer.sub_optimizer import SubOptimizerResult


@dataclass
class MeasurementNote:
    """Ein aus der Eingabe-JSON uebernommener REW-`Notes`-Text, z. B. die
    DSP-Einstellungen von Geraeten, die REW nicht automatisch auslesen kann
    (KEF, Saxx DS 120, ggf. Mosconi - siehe docs/REW_NOTES_TEMPLATE.md).

    Strikt getrennt von `Report.notes`: dort stehen die von der Pipeline
    selbst erzeugten System-/Warnhinweise, hier ausschliesslich der vom
    Nutzer in REW hinterlegte Klartext.
    """

    position_id: str
    channel: str | None  # "l" | "r" | "sub" | None (None = ganze Position)
    text: str


@dataclass
class Report:
    """Strukturierter Bericht einer Analyse-Session.

    Jede Teilmenge ist bewusst getrennt gehalten:
    - `measurement_summary`: reine Messwerte (z. B. gemittelte SPL-Kurven).
    - `mode_confirmations`: raeumlich/zeitlich bestaetigte Modenkandidaten
      (Modellbewertung einer Messung, kein reiner Messwert).
    - `sub_optimizer_result`: Ergebnis der Optimierung (Modellannahme
      70/30-Gewichtung, siehe sub_optimizer.py).
    - `eq_proposals`: Filterempfehlungen (immer FieldMeaning.MODEL).
    - `measurement_notes`: aus der Eingabe-JSON uebernommene REW-`Notes`-
      Texte (DSP-Einstellungen), siehe `MeasurementNote` oben - KEIN
      System-/Statushinweis.
    - `notes`: von der Pipeline selbst erzeugte Freitext-Hinweise (z. B.
      offene Punkte, Warnungen).
    """

    generated_at: str
    room_dimensions_m: tuple[float, float, float] | None
    measurement_summary: dict[str, Any] = field(default_factory=dict)
    mode_confirmations: list[ModeConfirmation] = field(default_factory=list)
    sub_optimizer_result: SubOptimizerResult | None = None
    eq_proposals_l: list[EqProposal] = field(default_factory=list)
    eq_proposals_r: list[EqProposal] = field(default_factory=list)
    measurement_notes: list[MeasurementNote] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Freigabestatus der EQ-Vorschlaege (None nur, wenn nie geprueft wurde -
    # wird im Report wie INCONCLUSIVE behandelt, nie als freigegeben).
    do_no_harm: NoHarmResult | None = None
    # Dritte Pruefebene: reale L+R-Summenmessung vs. komplexe Vorhersage
    # (None, wenn keine `lr`-Messung an sofa_mitte mitgegeben wurde).
    lr_sum_check: dict[str, Any] | None = None
    # Optional channel-specific confirmations; appended to preserve positional constructor compatibility.
    mode_confirmations_l: list[ModeConfirmation] | None = None
    mode_confirmations_r: list[ModeConfirmation] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "room_dimensions_m": self.room_dimensions_m,
            "messwerte": self.measurement_summary,
            "modellannahmen": {
                "mode_confirmations": [
                    {
                        "freq_hz": c.freq_hz,
                        "classification": c.classification,
                        "decay_confirmed": c.decay_confirmed,
                        "q_factor": c.q_factor,
                    }
                    for c in self.mode_confirmations
                ],
                "mode_confirmations_l": [
                    {"freq_hz": c.freq_hz, "classification": c.classification, "decay_confirmed": c.decay_confirmed, "q_factor": c.q_factor}
                    for c in (self.mode_confirmations_l if self.mode_confirmations_l is not None else self.mode_confirmations)
                ],
                "mode_confirmations_r": [
                    {"freq_hz": c.freq_hz, "classification": c.classification, "decay_confirmed": c.decay_confirmed, "q_factor": c.q_factor}
                    for c in (self.mode_confirmations_r if self.mode_confirmations_r is not None else self.mode_confirmations)
                ],
                "sub_optimizer": (
                    {
                        "best_gain_db": self.sub_optimizer_result.best_gain_db,
                        "best_delay_ms": self.sub_optimizer_result.best_delay_ms,
                        "best_polarity_deg": self.sub_optimizer_result.best_polarity_deg,
                        "best_score": self.sub_optimizer_result.best_score,
                        "mode_band_hz": self.sub_optimizer_result.mode_band_hz,
                        "mode_band_rmse_at_best_db": (
                            self.sub_optimizer_result.mode_band_rmse_at_best_db
                        ),
                    }
                    if self.sub_optimizer_result is not None
                    else None
                ),
            },
            "konsistenzpruefung": {"lr_summe": self.lr_sum_check},
            "empfehlungen": {
                "freigabe": release_banner(self.do_no_harm)[1],
                "freigabe_status": (
                    self.do_no_harm.status.value if self.do_no_harm else NoHarmStatus.INCONCLUSIVE.value
                ),
                "freigabe_grund": self.do_no_harm.reason if self.do_no_harm else "nicht geprueft",
                "eq_proposals_l": [_proposal_to_dict(p) for p in self.eq_proposals_l],
                "eq_proposals_r": [_proposal_to_dict(p) for p in self.eq_proposals_r],
            },
            "mess_und_dsp_notizen": [
                {"position_id": n.position_id, "channel": n.channel, "text": n.text}
                for n in self.measurement_notes
            ],
            "notes": self.notes,
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def write_html(self, path: str | Path) -> None:
        Path(path).write_text(render_html(self), encoding="utf-8")


def release_banner(result: NoHarmResult | None) -> tuple[str, str]:
    """(CSS-Klasse, Klartext) fuer den Freigabestatus. Alles ausser SAFE
    ist deutlich als NICHT FREIGEGEBEN gekennzeichnet."""
    if result is None:
        return "release-inconclusive", "UNGEPRUEFT / NICHT FREIGEGEBEN (Do-No-Harm-Pruefung nicht durchgefuehrt)"
    if result.status is NoHarmStatus.SAFE:
        return "release-safe", "SAFE - Do-No-Harm-Pruefung bestanden"
    if result.status is NoHarmStatus.UNSAFE:
        return "release-unsafe", "NICHT FREIGEGEBEN / UNSAFE - Schaden erkannt: " + result.reason
    return "release-inconclusive", "UNGEPRUEFT / NICHT FREIGEGEBEN (INCONCLUSIVE): " + result.reason


def _proposal_to_dict(p: EqProposal) -> dict[str, Any]:
    return {
        "type": p.filter.type,
        "fc_hz": p.filter.fc,
        "gain_db": p.filter.gain_db,
        "q": p.filter.q,
        "reason": p.reason,
        "meaning": p.meaning.value,
    }


def write_frequency_table_csv(
    path: str | Path, freq_hz: np.ndarray, columns: dict[str, np.ndarray]
) -> None:
    """Schreibt eine frequenzindizierte Tabelle (z. B. gemittelte SPL,
    Zielkurve, Streuung) als CSV."""
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    for name, values in columns.items():
        values = np.asarray(values)
        if values.shape != freq_hz.shape:
            raise ValueError(f"Spalte {name!r} hat eine andere Laenge als freq_hz")

    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["freq_hz", *columns.keys()])
        for i, f in enumerate(freq_hz):
            row = [f"{f:.4f}"]
            for values in columns.values():
                v = values[i]
                row.append("" if (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}")
            writer.writerow(row)


def render_html(report: Report) -> str:
    """Einfacher, abhaengigkeitsfreier HTML-Bericht (reines String-
    Templating, kein externes Template-Framework)."""
    e = html.escape

    def mode_rows(confirmations: list[ModeConfirmation]) -> str:
        rows = []
        for c in confirmations:
            decay = "unbekannt" if c.decay_confirmed is None else ("ja" if c.decay_confirmed else "nein")
            rows.append(
                f"<tr><td>{c.freq_hz:.1f} Hz</td><td>{e(c.classification)}</td>"
                f"<td>{decay}</td><td>{'' if c.q_factor is None else f'{c.q_factor:.2f}'}</td></tr>"
            )
        return "\n".join(rows) or "<tr><td colspan='4'><em>keine</em></td></tr>"

    confirmations_l = report.mode_confirmations_l if report.mode_confirmations_l is not None else report.mode_confirmations
    confirmations_r = report.mode_confirmations_r if report.mode_confirmations_r is not None else report.mode_confirmations

    def proposal_rows(proposals: list[EqProposal]) -> str:
        rows = []
        for p in proposals:
            rows.append(
                f"<tr><td>{e(p.filter.type)}</td><td>{p.filter.fc:.1f} Hz</td>"
                f"<td>{p.filter.gain_db:+.1f} dB</td><td>{p.filter.q:.2f}</td>"
                f"<td>{e(p.reason)}</td></tr>"
            )
        return "\n".join(rows) or "<tr><td colspan='5'><em>keine</em></td></tr>"

    sub = report.sub_optimizer_result
    sub_html = (
        f"<p>Pegel: {sub.best_gain_db:+.1f} dB, Delay: {sub.best_delay_ms:.1f} ms, "
        f"Polaritaet: {sub.best_polarity_deg:.0f} Grad, Score: {sub.best_score:.3f}<br>"
        f"RMSE im Modenband {sub.mode_band_hz[0]:.0f}-{sub.mode_band_hz[1]:.0f} Hz: "
        f"{sub.mode_band_rmse_at_best_db:.2f} dB "
        f"(<strong>Hinweis:</strong> Interferenzeffekt, nicht als behobene Mode interpretieren)</p>"
        if sub is not None
        else "<p><em>Keine Subwoofer-Optimierung in diesem Bericht.</em></p>"
    )

    def measurement_note_rows() -> str:
        rows = []
        channel_labels = {"l": "L", "r": "R", "sub": "Sub", "lr": "L+R"}
        for n in report.measurement_notes:
            channel_label = channel_labels.get(n.channel, "Position (gesamt)")
            rows.append(
                f"<tr><td>{e(n.position_id)}</td><td>{e(channel_label)}</td>"
                f"<td>{e(n.text)}</td></tr>"
            )
        return "\n".join(rows) or "<tr><td colspan='3'><em>keine</em></td></tr>"

    banner_class, banner_text = release_banner(report.do_no_harm)
    lr = report.lr_sum_check
    lr_html = (
        f"<p>Band {lr['band_hz'][0]:.0f}-{lr['band_hz'][1]:.0f} Hz, {lr['n_points']} Rasterpunkte: "
        f"RMSE {lr['rmse_db']:.2f} dB, max. |Differenz| {lr['max_abs_diff_db']:.2f} dB "
        f"(Vorhersage minus Messung).<br><strong>Hinweis:</strong> reiner Konsistenz-Check ohne "
        f"automatisches Urteil - grosse Abweichungen deuten auf verletzte Zeitbasis oder "
        f"Nichtlinearitaeten hin, nicht automatisch auf einen Messfehler.</p>"
        if lr is not None
        else "<p><em>Keine reale L+R-Summenmessung (<code>lr</code> an sofa_mitte) mitgegeben.</em></p>"
    )
    notes_html = "".join(f"<li>{e(n)}</li>" for n in report.notes) or "<li><em>keine</em></li>"

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Spatial Analyzer Bericht - {e(report.generated_at)}</title>
<style>
body {{ font-family: sans-serif; margin: 2em; color: #222; }}
h1, h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5em; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
.model-banner {{ background: #fff3cd; padding: 0.5em; border-radius: 4px; }}
.release-safe {{ background: #d4edda; padding: 0.6em; border-radius: 4px; font-weight: bold; }}
.release-unsafe {{ background: #f8d7da; padding: 0.6em; border-radius: 4px; font-weight: bold; }}
.release-inconclusive {{ background: #ffe5b4; padding: 0.6em; border-radius: 4px; font-weight: bold; }}
</style>
</head>
<body>
<h1>Spatial Analyzer Bericht</h1>
<p>Erstellt: {e(report.generated_at)}</p>
<p>Raummasse: {e(str(report.room_dimensions_m)) if report.room_dimensions_m else "unbekannt"} m
(<em>Modellwert fuer theoretische Modenkandidaten, kein Messwert</em>)</p>

<h2>Messwerte</h2>
<pre>{e(json.dumps(report.measurement_summary, indent=2, ensure_ascii=False))}</pre>

<h2>Mess- &amp; DSP-Notizen (aus REW)</h2>
<p><em>Vom Nutzer im REW-`Notes`-Feld hinterlegter Klartext (z. B. KEF-/Saxx-/Mosconi-
Einstellungen, die REW nicht automatisch auslesen kann - siehe
docs/REW_NOTES_TEMPLATE.md). Kein System-Hinweis der Pipeline, siehe
Abschnitt "Hinweise" unten dafuer.</em></p>
<table>
<tr><th>Position</th><th>Kanal</th><th>Notiz</th></tr>
{measurement_note_rows()}
</table>

<h2>Modellannahmen: Bestaetigte Modenkandidaten L</h2>
<div class="model-banner">Kanalgetrennte Modellbewertungen, keine direkten Messwerte.</div>
<table>
<tr><th>Frequenz</th><th>Klassifikation</th><th>Abkling-bestaetigt</th><th>Q</th></tr>
{mode_rows(confirmations_l)}
</table>
<h2>Modellannahmen: Bestaetigte Modenkandidaten R</h2>
<table>
<tr><th>Frequenz</th><th>Klassifikation</th><th>Abkling-bestaetigt</th><th>Q</th></tr>
{mode_rows(confirmations_r)}
</table>

<h2>Modellannahmen: Subwoofer-Optimierung</h2>
<div class="model-banner">70/30-Zielfunktion ist eine bewusst gewaehlte
Gewichtung, kein gemessener oder abgeleiteter Wert.</div>
{sub_html}

<h2>Konsistenzpruefung: L+R-Summe (dritte Pruefebene)</h2>
{lr_html}

<h2>Freigabestatus der EQ-Vorschlaege</h2>
<div class="{banner_class}">{e(banner_text)}</div>

<h2>Empfehlungen: EQ-Vorschlaege L</h2>
<table>
<tr><th>Typ</th><th>Fc</th><th>Gain</th><th>Q</th><th>Begruendung</th></tr>
{proposal_rows(report.eq_proposals_l)}
</table>

<h2>Empfehlungen: EQ-Vorschlaege R</h2>
<table>
<tr><th>Typ</th><th>Fc</th><th>Gain</th><th>Q</th><th>Begruendung</th></tr>
{proposal_rows(report.eq_proposals_r)}
</table>

<h2>Hinweise</h2>
<ul>
{notes_html}
</ul>
</body>
</html>
"""
