"""PEQ-Vorschlagslogik mit strikten "Do-No-Harm"-Sicherheitsregeln.

Siehe docs/SPECIFICATION.md Abschnitt 3.F und die Projekt-EQ-Regeln
(raum_akustik.md): im Modenbereich (20-200 Hz) ausschliesslich Cut-only-
Absenkungen, nur auf raeumlich konsistenten UND abklingbestaetigten
Raummoden (siehe bass_resonance.py/etc.py) - niemals auf wandernde
Kammfilter-Kerben oder unbestaetigte Einbrueche. Oberhalb 200 Hz begrenzte
Filterguete Q.

Jeder vorgeschlagene Filter wird vor der Aufnahme in die Ergebnisliste
gegen die Mehrpunkt-Vorschau (biquad_preview.py) geprueft
(validate_no_harm()). Das Ergebnis ist ein Drei-Stufen-Status
(SAFE/UNSAFE/INCONCLUSIVE, siehe NoHarmStatus): nur SAFE ist freigegeben;
bei UNSAFE/INCONCLUSIVE kennzeichnen pipeline.py und die Exporte die
Artefakte deutlich als NICHT FREIGEGEBEN.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from spatial_analyzer.biquad_preview import PeakingFilter, simulate_multi_position_preview
from spatial_analyzer.filter_digest import peaking_filters_digest
from spatial_analyzer.models import FieldMeaning

DEFAULT_MODE_BAND_HZ = (20.0, 200.0)
DEFAULT_MAX_MODE_CUT_DB = 9.0
DEFAULT_MAX_MODE_Q = 8.0
DEFAULT_MAX_BROADBAND_Q = 2.0
DEFAULT_MAX_BROADBAND_GAIN_DB = 6.0
DEFAULT_MIN_FC_SPACING_OCT = 0.5
DEFAULT_SPATIAL_HARM_TOLERANCE_DB = 0.5
DEFAULT_LOCAL_HARM_TOLERANCE_DB = 2.0  # Arbeitswert: max. Zunahme der Streuung an einer Frequenz
DEFAULT_TARGET_HARM_TOLERANCE_DB = 0.5  # Arbeitswert: max. Zunahme der RMS-Zielabweichung
MIN_POSITIONS_FOR_NO_HARM = 3  # Arbeitswert: darunter keine belastbare raeumliche Aussage
# Umfang der Pruefung: "left_right" = Vorschau ueber filters_l/filters_r ist gelaufen;
# "none" = keine (vollstaendige) Pruefung, z. B. keine Filter oder INCONCLUSIVE.
CHECKED_SCOPE_LEFT_RIGHT = "left_right"
CHECKED_SCOPE_NONE = "none"


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


class NoHarmStatus(str, Enum):
    """Drei-Stufen-Freigabestatus der Do-No-Harm-Pruefung.

    - SAFE: geprueft und innerhalb der Toleranzen.
    - UNSAFE: Schaden erkannt (globale oder frequenzlokale Zunahme der
      raeumlichen Streuung oder Verschlechterung der Zielkurvenabweichung).
    - INCONCLUSIVE: nicht belastbar pruefbar (z. B. keine Phase, weniger
      als `MIN_POSITIONS_FOR_NO_HARM` Positionen, nicht-endliche Werte).
      Wie UNSAFE ist das NICHT freigegeben - ein fehlendes Urteil ist
      kein "sicher".
    """

    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    INCONCLUSIVE = "INCONCLUSIVE"

    @property
    def is_released(self) -> bool:
        return self is NoHarmStatus.SAFE


@dataclass(frozen=True)
class NoHarmResult:
    """Ergebnis der Do-No-Harm-Pruefung.

    `checked_scope` und `checked_filters_sha256` sind KEIN Konstruktorargument
    (init=False): sie werden nur in validate_no_harm() gesetzt und belegen, dass
    die Vorschau tatsaechlich ueber die Filter lief. Ein selbst gebautes Ergebnis
    traegt scope "none" und sha None. Grenze: In-Prozess-Python kann eine gezielte
    Faelschung (z. B. object.__setattr__) nicht verhindern; die Bindung unterscheidet
    fuer normale Aufrufer eine durchgefuehrte von einer behaupteten Pruefung.
    """

    status: NoHarmStatus
    reason: str
    n_positions: int
    spread_before_db: float | None = None  # NaN-frei oder None
    spread_after_db: float | None = None
    max_local_spread_increase_db: float | None = None
    max_local_spread_increase_hz: float | None = None
    target_dev_before_db: float | None = None
    target_dev_after_db: float | None = None
    checked_scope: str = field(default=CHECKED_SCOPE_NONE, init=False)
    checked_filters_sha256: str | None = field(default=None, init=False)  # Digest der geprueften L/R-Filter
    # True nur, wenn validate_no_harm() mit den Standard-Toleranzen und
    # min_positions >= MIN_POSITIONS_FOR_NO_HARM lief (Schritt-3-Befund F1:
    # eine gelockerte Pruefung darf keine Freigabe begruenden).
    checked_default_tolerances: bool = field(default=False, init=False)
    # Zielkurvenkriterium ist tatsaechlich gelaufen (target_db uebergeben).
    checked_target_curve: bool = field(default=False, init=False)
    # Digest der geprueften Messdaten (freq, Positionen, L/R komplex, Ziel) -
    # bindet den Nachweis an die Messung (Schritt-3-Befund F2).
    checked_data_sha256: str | None = field(default=None, init=False)

    @property
    def is_released(self) -> bool:
        return self.status.is_released


def inconclusive_result(reason: str, n_positions: int = 0) -> NoHarmResult:
    return NoHarmResult(NoHarmStatus.INCONCLUSIVE, reason, n_positions)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def validate_no_harm(
    freq_hz: np.ndarray,
    position_ids: list[str],
    l_complex: np.ndarray,
    r_complex: np.ndarray,
    filters_l: list[PeakingFilter],
    filters_r: list[PeakingFilter],
    fs: float = 48000.0,
    tolerance_db: float = DEFAULT_SPATIAL_HARM_TOLERANCE_DB,
    local_tolerance_db: float = DEFAULT_LOCAL_HARM_TOLERANCE_DB,
    target_db: np.ndarray | None = None,
    target_tolerance_db: float = DEFAULT_TARGET_HARM_TOLERANCE_DB,
    min_positions: int = MIN_POSITIONS_FOR_NO_HARM,
) -> NoHarmResult:
    """Prueft, ob die vorgeschlagenen Filter die raeumliche Streuung (Std.
    ueber alle Positionen) der L+R-Summe oder die Zielkurvenabweichung
    verschlechtern, und liefert ein explizites Drei-Stufen-Urteil
    (`NoHarmStatus`) statt eines blossen `bool`.

    Kriterien (alle Toleranzen sind Arbeitswerte, keine physikalischen
    Grenzen):
    1. Globale Streuung: Mittel der Std. ueber alle Frequenzen
       (nachher - vorher > `tolerance_db` -> UNSAFE).
    2. Frequenzlokal/schmalbandig: groesste Zunahme der Std. an EINER
       Frequenz (> `local_tolerance_db` -> UNSAFE) - ein globaler Mittelwert
       kann eine schmalbandige Verschlechterung verdecken.
    3. Zielkurve (nur wenn `target_db` auf dem Raster von `freq_hz`
       uebergeben wird): RMS-Abweichung des positionsgemittelten L- bzw.
       R-Pegels zur Zielkurve; Zunahme > `target_tolerance_db` -> UNSAFE.

    INCONCLUSIVE (statt eines stillen NaN-Vergleichs): weniger als
    `min_positions` Positionen (Std. mit ddof=1 ist bei N=1 NaN, bei N=2
    keine belastbare raeumliche Aussage), nicht-endliche Eingaben oder
    nicht-endliche Zwischenwerte. Ohne Filter gibt es nichts zu
    verschlechtern: SAFE.
    """
    n_positions = int(np.asarray(l_complex).shape[0]) if np.ndim(l_complex) == 2 else 0
    if not filters_l and not filters_r:
        return NoHarmResult(
            NoHarmStatus.SAFE, "keine Filter vorgeschlagen - nichts zu verschlechtern", n_positions
        )
    if n_positions < min_positions:
        return inconclusive_result(
            f"nur {n_positions} Position(en) - fuer eine belastbare raeumliche Aussage "
            f"sind mindestens {min_positions} noetig",
            n_positions,
        )

    preview = simulate_multi_position_preview(
        freq_hz, position_ids, l_complex, r_complex, filters_l, filters_r, fs
    )
    std_before = np.std(preview.lr_before_db, axis=0, ddof=1)
    std_after = np.std(preview.lr_after_db, axis=0, ddof=1)
    if not (np.all(np.isfinite(std_before)) and np.all(np.isfinite(std_after))):
        return inconclusive_result("nicht-endliche Streuungswerte in der Vorschau", n_positions)

    spread_before = float(np.mean(std_before))
    spread_after = float(np.mean(std_after))
    delta = std_after - std_before
    worst = int(np.argmax(delta))
    max_local = float(delta[worst])
    max_local_hz = float(np.asarray(freq_hz, dtype=np.float64)[worst])

    dev_before = dev_after = None
    if target_db is not None:
        target = np.asarray(target_db, dtype=np.float64)
        if target.shape != np.asarray(freq_hz).shape or not np.all(np.isfinite(target)):
            return inconclusive_result("target_db passt nicht zu freq_hz oder ist nicht endlich", n_positions)
        dev_before = max(
            _rms(np.mean(preview.l_before_db, axis=0) - target),
            _rms(np.mean(preview.r_before_db, axis=0) - target),
        )
        dev_after = max(
            _rms(np.mean(preview.l_after_db, axis=0) - target),
            _rms(np.mean(preview.r_after_db, axis=0) - target),
        )

    problems = []
    if spread_after - spread_before > tolerance_db:
        problems.append(
            f"mittlere raeumliche Streuung steigt {spread_before:.2f} dB -> {spread_after:.2f} dB"
        )
    if max_local > local_tolerance_db:
        problems.append(
            f"lokale Streuung steigt um {max_local:.2f} dB bei {max_local_hz:.1f} Hz"
        )
    if dev_before is not None and dev_after - dev_before > target_tolerance_db:
        problems.append(
            f"Zielkurvenabweichung steigt {dev_before:.2f} dB -> {dev_after:.2f} dB"
        )

    if problems:
        status, reason = NoHarmStatus.UNSAFE, "; ".join(problems)
    else:
        status = NoHarmStatus.SAFE
        reason = f"raeumliche Streuung {spread_before:.2f} dB -> {spread_after:.2f} dB, innerhalb der Toleranzen"
    result = NoHarmResult(
        status, reason, n_positions, spread_before, spread_after, max_local, max_local_hz,
        dev_before, dev_after,
    )
    # Frozen dataclass und init=False: Bindung nur hier, direkt nach dem Erzeugen (siehe Klassen-Docstring).
    object.__setattr__(result, "checked_scope", CHECKED_SCOPE_LEFT_RIGHT)
    object.__setattr__(result, "checked_filters_sha256", peaking_filters_digest(filters_l, filters_r))
    object.__setattr__(
        result,
        "checked_default_tolerances",
        tolerance_db == DEFAULT_SPATIAL_HARM_TOLERANCE_DB
        and local_tolerance_db == DEFAULT_LOCAL_HARM_TOLERANCE_DB
        and target_tolerance_db == DEFAULT_TARGET_HARM_TOLERANCE_DB
        and min_positions >= MIN_POSITIONS_FOR_NO_HARM,
    )
    object.__setattr__(result, "checked_target_curve", target_db is not None)
    object.__setattr__(
        result, "checked_data_sha256",
        measurement_data_digest(freq_hz, position_ids, l_complex, r_complex, target_db),
    )
    return result


def measurement_data_digest(
    freq_hz: np.ndarray,
    position_ids: list[str],
    l_complex: np.ndarray,
    r_complex: np.ndarray,
    target_db: np.ndarray | None,
) -> str:
    """SHA-256 ueber die Daten einer Do-No-Harm-Pruefung (deterministisch;
    -0.0 wird zu 0.0 normiert, Formen und Positions-IDs gehen mit ein)."""
    h = hashlib.sha256(b"spatial_analyzer.no_harm_data.v1\0")

    def _arr(x: np.ndarray | None, dtype: type) -> None:
        if x is None:
            h.update(b"none\0")
            return
        a = np.ascontiguousarray(np.asarray(x, dtype=dtype) + 0.0)  # -0.0 -> 0.0
        h.update(repr(a.shape).encode() + b"\0" + a.tobytes())

    _arr(freq_hz, np.float64)
    h.update(len(position_ids).to_bytes(8, "big"))
    for pid in position_ids:  # laengenpraefixiert: keine Kollision durch Trennzeichen in IDs
        b = str(pid).encode("utf-8")
        h.update(len(b).to_bytes(8, "big") + b)
    _arr(l_complex, np.complex128)
    _arr(r_complex, np.complex128)
    _arr(target_db, np.float64)
    return h.hexdigest()
