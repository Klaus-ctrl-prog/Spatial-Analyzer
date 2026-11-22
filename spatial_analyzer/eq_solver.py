"""PEQ-Vorschlagslogik mit strikten "Do-No-Harm"-Sicherheitsregeln.

Siehe docs/SPECIFICATION.md Abschnitt 3.F und die Projekt-EQ-Regeln
(raum_akustik.md): im Modenbereich (20-200 Hz) ausschliesslich Cut-only-
Absenkungen, nur auf raeumlich konsistenten UND abklingbestaetigten
Raummoden (siehe bass_resonance.py/etc.py) - niemals auf wandernde
Kammfilter-Kerben oder unbestaetigte Einbrueche. Oberhalb 200 Hz begrenzte
Filterguete Q.

Jeder vorgeschlagene Filter wird vor der Aufnahme in die Ergebnisliste
gegen die Mehrpunkt-Vorschau (biquad_preview.py) geprueft: verschlechtert
er die raeumliche Streuung ueber die Positionen merklich, wird er verworfen
statt blind uebernommen (validate_no_harm()).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatial_analyzer.biquad_preview import PeakingFilter, simulate_multi_position_preview
from spatial_analyzer.models import FieldMeaning

DEFAULT_MODE_BAND_HZ = (20.0, 200.0)
DEFAULT_MAX_MODE_CUT_DB = 9.0
DEFAULT_MAX_MODE_Q = 8.0
DEFAULT_MAX_BROADBAND_Q = 2.0
DEFAULT_MAX_BROADBAND_GAIN_DB = 6.0
DEFAULT_MIN_FC_SPACING_OCT = 0.5
DEFAULT_SPATIAL_HARM_TOLERANCE_DB = 0.5


@dataclass
class ModeConfirmation:
    """Ein per bass_resonance.py/etc.py bestaetigter Modenkandidat.

    `classification` stammt aus etc.classify_bass_dip() ("mode_candidate"
    oder "mode_candidate_unmodeled"); `decay_confirmed` aus
    bass_resonance.evaluate_decay_against_threshold().exceeds_perceptual_threshold.
    Nur wenn BEIDES zutrifft, wird ein Cut vorgeschlagen (siehe Modul-
    Docstring: "raeumlich konsistent UND abklingbestaetigt").
    """

    freq_hz: float
    classification: str
    decay_confirmed: bool | None
    q_factor: float | None = None


@dataclass
class EqProposal:
    filter: PeakingFilter
    reason: str
    meaning: FieldMeaning = FieldMeaning.MODEL  # der Filter ist ein Vorschlag, kein Messwert


def propose_mode_cuts(
    freq_hz: np.ndarray,
    averaged_spl_db: np.ndarray,
    target_spl_db: np.ndarray,
    confirmations: list[ModeConfirmation],
    band_hz: tuple[float, float] = DEFAULT_MODE_BAND_HZ,
    max_cut_db: float = DEFAULT_MAX_MODE_CUT_DB,
    max_q: float = DEFAULT_MAX_MODE_Q,
    default_q: float = 4.0,
) -> list[EqProposal]:
    """Cut-only PEQ-Vorschlaege fuer bestaetigte Raummoden im Modenbereich.

    Nur Confirmations mit classification in
    ("mode_candidate", "mode_candidate_unmodeled") UND decay_confirmed is
    True werden beruecksichtigt - alles andere (Kammfilter-Kandidaten,
    unbestaetigte oder nicht abklingbestaetigte Einbrueche) wird bewusst
    uebersprungen, auch wenn an der Stelle ein messbarer Peak vorliegt.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    averaged_spl_db = np.asarray(averaged_spl_db, dtype=np.float64)
    target_spl_db = np.asarray(target_spl_db, dtype=np.float64)

    proposals: list[EqProposal] = []
    for conf in confirmations:
        if not (band_hz[0] <= conf.freq_hz <= band_hz[1]):
            continue
        if conf.classification not in ("mode_candidate", "mode_candidate_unmodeled"):
            continue
        if conf.decay_confirmed is not True:
            continue

        idx = int(np.argmin(np.abs(freq_hz - conf.freq_hz)))
        excess_db = averaged_spl_db[idx] - target_spl_db[idx]
        if excess_db <= 0:
            continue  # bereits auf oder unter Zielkurve, kein Cut noetig

        gain_db = -min(excess_db, max_cut_db)
        q = min(conf.q_factor, max_q) if conf.q_factor else default_q
        proposals.append(
            EqProposal(
                filter=PeakingFilter(type="peq", fc=conf.freq_hz, gain_db=gain_db, q=q),
                reason=(
                    f"Raeumlich konsistente, abklingbestaetigte Mode bei {conf.freq_hz:.1f} Hz "
                    f"({excess_db:.1f} dB ueber Zielkurve) - Cut-only."
                ),
            )
        )
    return proposals


def propose_broadband_correction(
    freq_hz: np.ndarray,
    averaged_spl_db: np.ndarray,
    target_spl_db: np.ndarray,
    band_hz: tuple[float, float] = (200.0, 20000.0),
    max_q: float = DEFAULT_MAX_BROADBAND_Q,
    max_boost_db: float = DEFAULT_MAX_BROADBAND_GAIN_DB,
    max_cut_db: float = DEFAULT_MAX_BROADBAND_GAIN_DB,
    min_fc_spacing_oct: float = DEFAULT_MIN_FC_SPACING_OCT,
    max_filters: int = 6,
    deviation_threshold_db: float = 1.0,
    excluded_freqs_hz: list[float] | None = None,
) -> list[EqProposal]:
    """Breitbandige Korrektur oberhalb des Modenbereichs: greedy Auswahl der
    groessten Abweichungen zur Zielkurve, mit begrenzter Guete (max_q) und
    begrenztem Gain, Mindestabstand zwischen Filtern (Oktaven) und
    Ausschluss bekannter Kammfilter-Frequenzen (`excluded_freqs_hz`, z. B.
    aus etc.classify_bass_dip()=="comb_filter_candidate" - Boosts werden
    dort NICHT vorgeschlagen, siehe Projekt-EQ-Regeln).
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    averaged_spl_db = np.asarray(averaged_spl_db, dtype=np.float64)
    target_spl_db = np.asarray(target_spl_db, dtype=np.float64)
    excluded_freqs_hz = excluded_freqs_hz or []

    band_mask = (freq_hz >= band_hz[0]) & (freq_hz <= band_hz[1])
    deviation = np.full_like(averaged_spl_db, np.nan)
    deviation[band_mask] = averaged_spl_db[band_mask] - target_spl_db[band_mask]

    chosen_fc_log: list[float] = []
    proposals: list[EqProposal] = []

    working = deviation.copy()
    for _ in range(max_filters):
        with np.errstate(invalid="ignore"):
            idx = np.nanargmax(np.abs(working)) if np.any(~np.isnan(working)) else None
        if idx is None:
            break
        dev = working[idx]
        if np.isnan(dev) or abs(dev) < deviation_threshold_db:
            break
        fc = float(freq_hz[idx])

        too_close = any(abs(np.log2(fc / f)) < min_fc_spacing_oct for f in chosen_fc_log)
        is_excluded = any(abs(fc - ex) < 1.0 for ex in excluded_freqs_hz)
        if too_close or is_excluded:
            working[idx] = np.nan
            continue

        gain_db = -float(np.clip(dev, -max_cut_db, max_boost_db))
        proposals.append(
            EqProposal(
                filter=PeakingFilter(type="peq", fc=fc, gain_db=gain_db, q=max_q),
                reason=f"Groesste Abweichung zur Zielkurve bei {fc:.0f} Hz ({dev:+.1f} dB).",
            )
        )
        chosen_fc_log.append(fc)
        # Umliegenden Bereich um fc aus der weiteren Suche ausschliessen,
        # damit nicht mehrere Filter auf denselben breiten Buckel gesetzt
        # werden.
        exclude_mask = np.abs(np.log2(np.maximum(freq_hz, 1e-9) / fc)) < min_fc_spacing_oct
        working[exclude_mask] = np.nan

    return proposals


def validate_no_harm(
    freq_hz: np.ndarray,
    position_ids: list[str],
    l_complex: np.ndarray,
    r_complex: np.ndarray,
    filters_l: list[PeakingFilter],
    filters_r: list[PeakingFilter],
    fs: float = 48000.0,
    tolerance_db: float = DEFAULT_SPATIAL_HARM_TOLERANCE_DB,
) -> tuple[bool, float, float]:
    """Prueft, ob die vorgeschlagenen Filter die raeumliche Streuung (Std.
    ueber alle Positionen, gemittelt ueber den Frequenzbereich) der L+R-Summe
    nicht merklich verschlechtern.

    Rueckgabe: (ist_sicher, streuung_vorher_db, streuung_nachher_db).
    """
    preview = simulate_multi_position_preview(
        freq_hz, position_ids, l_complex, r_complex, filters_l, filters_r, fs
    )
    spread_before = float(np.mean(np.std(preview.lr_before_db, axis=0, ddof=1)))
    spread_after = float(np.mean(np.std(preview.lr_after_db, axis=0, ddof=1)))
    is_safe = (spread_after - spread_before) <= tolerance_db
    return is_safe, spread_before, spread_after
