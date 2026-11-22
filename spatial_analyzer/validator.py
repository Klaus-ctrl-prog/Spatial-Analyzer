"""Schema-/Plausibilitaetspruefung der Eingabe-JSON vor Rechenbeginn
("--dry-run"), siehe cli.py.

Ziel: fehlende oder falsch benannte Felder in der von Hand gebauten
JSON-Zwischenform (siehe pipeline.py-Docstring) im Klartext melden, statt
dass `run_pipeline()` mit einem schwer lesbaren Python-Traceback abbricht.

Diese Pruefung bildet absichtlich nur die Mindestanforderungen von
`pipeline.py` nach (siehe dortige `_interpolate_channel_to_common_grid` und
`run_pipeline`), erfindet aber keine zusaetzlichen Pflichtfelder, die die
Pipeline selbst nicht braucht. Plausibilitaetsgrenzen (z. B. SPL-Bereich)
sind bewusst grosszuegige Arbeitswerte, keine physikalisch harten Grenzen -
sie sollen offensichtliche Tippfehler fangen (z. B. dB statt linear, Hz
vertauscht), nicht ungewoehnliche, aber reale Messungen blockieren.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLAUSIBLE_FREQ_RANGE_HZ = (1.0, 40000.0)
PLAUSIBLE_SPL_RANGE_DB = (-20.0, 140.0)
PLAUSIBLE_SAMPLE_INTERVAL_S = (1e-7, 1.0)


@dataclass
class ValidationIssue:
    """Ein einzelner Befund. `severity="error"` bedeutet: `run_pipeline()`
    wuerde an dieser Stelle mit hoher Wahrscheinlichkeit abbrechen oder ein
    unsinniges Ergebnis liefern. `severity="warning"` bedeutet: lauffaehig,
    aber auffaellig (z. B. keine Referenzposition `sofa_mitte`)."""

    severity: str  # "error" | "warning"
    path: str  # z. B. "positions[2].l.spl_db"
    message: str

    def __str__(self) -> str:
        label = "FEHLER" if self.severity == "error" else "Hinweis"
        return f"[{label}] {self.path}: {self.message}"


def _check_channel(
    position_index: int, position_id: str, channel_key: str, channel: Any
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    prefix = f"positions[{position_index}] (id={position_id!r}).{channel_key}"

    if not isinstance(channel, dict):
        issues.append(ValidationIssue("error", prefix, "muss ein Objekt sein"))
        return issues

    freq = channel.get("freq_hz")
    spl = channel.get("spl_db")
    if freq is None:
        issues.append(ValidationIssue("error", f"{prefix}.freq_hz", "fehlt"))
    if spl is None:
        issues.append(ValidationIssue("error", f"{prefix}.spl_db", "fehlt"))
    if freq is None or spl is None:
        return issues

    if not isinstance(freq, list) or not isinstance(spl, list):
        issues.append(ValidationIssue("error", prefix, "freq_hz und spl_db muessen Listen sein"))
        return issues

    if len(freq) == 0:
        issues.append(ValidationIssue("error", f"{prefix}.freq_hz", "ist leer"))
    if len(freq) != len(spl):
        issues.append(
            ValidationIssue(
                "error", prefix, f"freq_hz (Laenge {len(freq)}) und spl_db (Laenge {len(spl)}) "
                "muessen gleich lang sein"
            )
        )

    phase = channel.get("phase_deg")
    if phase is not None:
        if not isinstance(phase, list):
            issues.append(ValidationIssue("error", f"{prefix}.phase_deg", "muss eine Liste sein"))
        elif len(phase) != len(freq):
            issues.append(
                ValidationIssue(
                    "error", f"{prefix}.phase_deg",
                    f"Laenge {len(phase)} passt nicht zu freq_hz (Laenge {len(freq)})",
                )
            )

    if len(freq) and len(freq) == len(spl):
        try:
            freq_f = [float(x) for x in freq]
            spl_f = [float(x) for x in spl]
        except (TypeError, ValueError):
            issues.append(ValidationIssue("error", prefix, "freq_hz/spl_db enthalten nicht-numerische Werte"))
        else:
            # Haertung (Prioritaet 5): NaN/Inf werden VOR der Monotonie- und
            # Plausibilitaetspruefung explizit als "error" erkannt, nicht
            # erst beilaeufig als "warning" ueber die Plausibilitaetsgrenzen
            # mitgefangen. Grund: `float('nan') <= f1` ist immer `False`,
            # die Monotonie-Pruefung wuerde ein NaN also unbemerkt
            # passieren lassen - und genau dieses stille Durchrutschen hat
            # sich in `FrequencyGrid`/`interpolate_to_grid` als echter
            # Fund bestaetigt (siehe grid.py/models.py, selbe Haertungsrunde).
            non_finite_freq = [f for f in freq_f if not math.isfinite(f)]
            if non_finite_freq:
                issues.append(
                    ValidationIssue(
                        "error", f"{prefix}.freq_hz",
                        f"{len(non_finite_freq)} Wert(e) sind NaN/Inf - run_pipeline() "
                        "wuerde dies nicht abfangen (FrequencyGrid/interpolate_to_grid "
                        "lehnen NaN/Inf explizit ab)",
                    )
                )
            non_finite_spl = [s for s in spl_f if not math.isfinite(s)]
            if non_finite_spl:
                issues.append(
                    ValidationIssue(
                        "error", f"{prefix}.spl_db",
                        f"{len(non_finite_spl)} Wert(e) sind NaN/Inf",
                    )
                )
            if not non_finite_freq and any(f2 <= f1 for f1, f2 in zip(freq_f, freq_f[1:])):
                issues.append(
                    ValidationIssue(
                        "warning", f"{prefix}.freq_hz",
                        "ist nicht streng monoton steigend - interpolate_to_grid() wird dies als Fehler ablehnen",
                    )
                )
            finite_freq = [f for f in freq_f if math.isfinite(f)]
            out_of_range_freq = [
                f for f in finite_freq if not (PLAUSIBLE_FREQ_RANGE_HZ[0] <= f <= PLAUSIBLE_FREQ_RANGE_HZ[1])
            ]
            if out_of_range_freq:
                issues.append(
                    ValidationIssue(
                        "warning", f"{prefix}.freq_hz",
                        f"{len(out_of_range_freq)} Werte ausserhalb des plausiblen Bereichs "
                        f"{PLAUSIBLE_FREQ_RANGE_HZ} Hz (z. B. {out_of_range_freq[0]}) - Tippfehler?",
                    )
                )
            finite_spl = [s for s in spl_f if math.isfinite(s)]
            out_of_range_spl = [
                s for s in finite_spl if not (PLAUSIBLE_SPL_RANGE_DB[0] <= s <= PLAUSIBLE_SPL_RANGE_DB[1])
            ]
            if out_of_range_spl:
                issues.append(
                    ValidationIssue(
                        "warning", f"{prefix}.spl_db",
                        f"{len(out_of_range_spl)} Werte ausserhalb des plausiblen Bereichs "
                        f"{PLAUSIBLE_SPL_RANGE_DB} dB (z. B. {out_of_range_spl[0]}) - Tippfehler "
                        "oder falsche Einheit?",
                    )
                )

    ir = channel.get("impulse_response")
    if ir is not None:
        if not isinstance(ir, dict):
            issues.append(ValidationIssue("error", f"{prefix}.impulse_response", "muss ein Objekt sein"))
        else:
            samples = ir.get("samples")
            dt = ir.get("sample_interval_s")
            if not isinstance(samples, list) or len(samples) == 0:
                issues.append(
                    ValidationIssue("error", f"{prefix}.impulse_response.samples", "fehlt oder ist leer")
                )
            elif not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in samples):
                # Haertung (Prioritaet 5): ohne diese Pruefung wuerde ein
                # nicht-numerisches Sample (z. B. versehentlich ein String)
                # hier unbemerkt bleiben und erst in
                # ImpulseResponseData.__post_init__ / np.asarray(...,
                # dtype=float64) mit einem rohen ValueError abbrechen.
                issues.append(
                    ValidationIssue(
                        "error", f"{prefix}.impulse_response.samples",
                        "enthaelt nicht-numerische Werte",
                    )
                )
            if not isinstance(dt, (int, float)) or isinstance(dt, bool):
                issues.append(
                    ValidationIssue("error", f"{prefix}.impulse_response.sample_interval_s", "fehlt oder ist keine Zahl")
                )
            elif not math.isfinite(dt):
                issues.append(
                    ValidationIssue(
                        "error", f"{prefix}.impulse_response.sample_interval_s",
                        "ist NaN/Inf - ImpulseResponseData lehnt das explizit ab",
                    )
                )
            elif not (PLAUSIBLE_SAMPLE_INTERVAL_S[0] <= dt <= PLAUSIBLE_SAMPLE_INTERVAL_S[1]):
                issues.append(
                    ValidationIssue(
                        "warning", f"{prefix}.impulse_response.sample_interval_s",
                        f"Wert {dt} ausserhalb des plausiblen Bereichs {PLAUSIBLE_SAMPLE_INTERVAL_S} s "
                        "(entspricht Samplerate ca. 1 Hz - 10 MHz) - vertauscht mit einer Frequenz?",
                    )
                )

    note = channel.get("notes")
    if note is not None and not isinstance(note, str):
        issues.append(ValidationIssue("warning", f"{prefix}.notes", "ist kein Text - wird ignoriert"))

    return issues


def validate_input(data: Any) -> list[ValidationIssue]:
    """Prueft die bereits als Python-Objekt vorliegende Eingabestruktur
    (siehe pipeline.py-Docstring fuer das Schema) und gibt eine Liste von
    Befunden zurueck (leer = keine Beanstandungen)."""
    issues: list[ValidationIssue] = []

    if not isinstance(data, dict):
        return [ValidationIssue("error", "<root>", "Eingabe muss ein JSON-Objekt sein")]

    room_dims = data.get("room_dimensions_m")
    if room_dims is not None:
        if not isinstance(room_dims, list) or len(room_dims) != 3:
            issues.append(
                ValidationIssue(
                    "error", "room_dimensions_m", "muss eine Liste mit genau 3 Zahlen [laenge, breite, hoehe] sein"
                )
            )
        else:
            try:
                dims_f = [float(x) for x in room_dims]
            except (TypeError, ValueError):
                issues.append(ValidationIssue("error", "room_dimensions_m", "enthaelt nicht-numerische Werte"))
            else:
                if any(d <= 0 for d in dims_f):
                    issues.append(ValidationIssue("error", "room_dimensions_m", "alle Werte muessen positiv sein"))

    positions = data.get("positions")
    if positions is None:
        issues.append(ValidationIssue("error", "positions", "fehlt"))
        return issues
    if not isinstance(positions, list) or len(positions) == 0:
        issues.append(ValidationIssue("error", "positions", "muss eine nicht-leere Liste sein"))
        return issues

    seen_ids: dict[str, int] = {}
    has_l = has_r = has_sofa_mitte = False

    for i, position in enumerate(positions):
        prefix = f"positions[{i}]"
        if not isinstance(position, dict):
            issues.append(ValidationIssue("error", prefix, "muss ein Objekt sein"))
            continue

        pos_id = position.get("id")
        if not isinstance(pos_id, str) or not pos_id.strip():
            issues.append(ValidationIssue("error", f"{prefix}.id", "fehlt oder ist leer"))
            pos_id = f"<position {i}>"
        else:
            if pos_id in seen_ids:
                issues.append(
                    ValidationIssue(
                        "warning", f"{prefix}.id",
                        f"doppelte id {pos_id!r} (bereits bei positions[{seen_ids[pos_id]}])",
                    )
                )
            seen_ids[pos_id] = i
            if pos_id == "sofa_mitte":
                has_sofa_mitte = True

        if "l" not in position and "r" not in position:
            issues.append(ValidationIssue("error", prefix, "hat weder Kanal 'l' noch 'r'"))

        for channel_key in ("l", "r", "sub"):
            if channel_key in position:
                if channel_key == "l":
                    has_l = True
                if channel_key == "r":
                    has_r = True
                issues.extend(_check_channel(i, pos_id, channel_key, position[channel_key]))

        pos_note = position.get("notes")
        if pos_note is not None and not isinstance(pos_note, str):
            issues.append(ValidationIssue("warning", f"{prefix}.notes", "ist kein Text - wird ignoriert"))

    if not has_l:
        issues.append(
            ValidationIssue("error", "positions", "keine einzige Position hat einen Kanal 'l' - run_pipeline() "
                             "wuerde mit 'Keine Positionen mit Kanal l gefunden' abbrechen")
        )
    if not has_r:
        issues.append(
            ValidationIssue("error", "positions", "keine einzige Position hat einen Kanal 'r' - run_pipeline() "
                             "wuerde mit 'Keine Positionen mit Kanal r gefunden' abbrechen")
        )
    if not has_sofa_mitte:
        issues.append(
            ValidationIssue(
                "warning", "positions",
                "keine Position mit id='sofa_mitte' gefunden - die erste Position wird als Referenz "
                "fuer die Abklinganalyse verwendet (siehe Bericht-Hinweise)",
            )
        )
    if len(positions) < 4:
        issues.append(
            ValidationIssue(
                "warning", "positions",
                f"nur {len(positions)} Position(en) - fuer eine belastbare raeumliche Konsistenzpruefung "
                "sind laut Messregeln 8-12 'Wedel'-Punkte plus Sofamitte vorgesehen",
            )
        )

    return issues


def validate_file(path: str | Path) -> list[ValidationIssue]:
    """Laedt und prueft eine Eingabedatei. Ein JSON-Parse-Fehler oder eine
    nicht lesbare Datei wird als einzelner `error`-Befund zurueckgegeben,
    nicht als Exception weitergereicht - der Zweck dieser Funktion ist
    gerade, Klartext statt Traceback zu liefern."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [ValidationIssue("error", str(path), f"Datei nicht lesbar: {exc}")]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [ValidationIssue("error", str(path), f"kein gueltiges JSON: {exc}")]

    return validate_input(data)


def has_errors(issues: list[ValidationIssue]) -> bool:
    return any(i.severity == "error" for i in issues)
