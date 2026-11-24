"""Energetische Mittelung (Variante A & B), Streuungsmasse, Case-Bootstrap.

Siehe docs/SPECIFICATION.md Abschnitt 3.C und Zwischenstand-Doku Abschnitt 3/7a.

Variante A: absoluter energetischer Mittelwert ohne Pegelnormalisierung.
Variante B: vorherige Pegelangleichung im Referenzband (Default 400 Hz-2 kHz,
analog REW "Align SPL"), danach energetische Mittelung.

Alle Funktionen arbeiten auf einer (n_positions, n_frequencies)-Matrix in dB
und sind NaN-sicher (fehlende/ungueltige Werte, z. B. ausserhalb des
R160-Ueberlappungsbereichs, werden ignoriert statt ersetzt - siehe grid.py).

Case-Bootstrap nach Efron (1979): Resampling mit Zuruecklegen ueber die
Positionen (Zeilen), nicht ueber die Frequenzachse. Explizit KEINE SVD
(siehe Zwischenstand-Doku Abschnitt 7a - Korrektur der urspruenglichen
Annahme). Bei kleinem/raeumlich korreliertem N (hier 11 Positionen, davon
10 benachbarte Wanderpunkte um dieselbe Sofamitte) sind die Baender mit
Vorsicht zu interpretieren - das wird hier als Modellannahme dokumentiert,
nicht verschwiegen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_ALIGN_BAND_HZ = (400.0, 2000.0)
DEFAULT_BOOTSTRAP_RESAMPLES = 500
DEFAULT_BOOTSTRAP_ALPHA = 0.10  # -> 90%-Band


def _validate_matrix(values_db: np.ndarray) -> np.ndarray:
    values_db = np.asarray(values_db, dtype=np.float64)
    if values_db.ndim != 2:
        raise ValueError("values_db muss eine (n_positions, n_frequencies)-Matrix sein")
    if values_db.shape[0] < 1:
        raise ValueError("values_db braucht mindestens eine Position (Zeile)")
    return values_db


def energetic_mean_db(values_db: np.ndarray, axis: int = 0) -> np.ndarray:
    """L_avg = 10*log10(mean(10**(L/10))) entlang `axis`, NaN-sicher.

    Entspricht REWs "RMS Average" / "Average The Responses" (verifiziert,
    siehe Zwischenstand-Doku Abschnitt 3): inkohaerente energetische
    Mittelung, keine Phaseninformation.
    """
    values_db = np.asarray(values_db, dtype=np.float64)
    linear_power = 10.0 ** (values_db / 10.0)
    mean_power = np.nanmean(linear_power, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 10.0 * np.log10(mean_power)


def average_variant_a(values_db: np.ndarray) -> np.ndarray:
    """Variante A: absoluter energetischer Mittelwert, keine Normalisierung."""
    values_db = _validate_matrix(values_db)
    return energetic_mean_db(values_db, axis=0)


def align_spl(
    values_db: np.ndarray,
    freqs_hz: np.ndarray,
    align_band_hz: tuple[float, float] = DEFAULT_ALIGN_BAND_HZ,
) -> np.ndarray:
    """Pegelangleichung je Position (Zeile) auf den energetischen Mittelwert
    im Referenzband (Default 400 Hz - 2 kHz), analog REWs "Align SPL".

    Jede Zeile wird um die Differenz zwischen ihrem eigenen Referenzband-
    Mittelwert und dem ueber alle Positionen gemeinsamen Referenzband-
    Mittelwert verschoben, sodass alle Positionen anschliessend denselben
    Referenzpegel haben.
    """
    values_db = _validate_matrix(values_db)
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    if freqs_hz.shape[0] != values_db.shape[1]:
        raise ValueError("freqs_hz muss zur Spaltenzahl von values_db passen")

    band_mask = (freqs_hz >= align_band_hz[0]) & (freqs_hz <= align_band_hz[1])
    if not np.any(band_mask):
        raise ValueError(f"Referenzband {align_band_hz} Hz liegt ausserhalb von freqs_hz")

    per_position_ref = energetic_mean_db(values_db[:, band_mask], axis=1)
    common_ref = float(np.nanmean(per_position_ref))
    offsets = common_ref - per_position_ref
    return values_db + offsets[:, np.newaxis]


def average_variant_b(
    values_db: np.ndarray,
    freqs_hz: np.ndarray,
    align_band_hz: tuple[float, float] = DEFAULT_ALIGN_BAND_HZ,
) -> np.ndarray:
    """Variante B: Pegelangleichung im Referenzband, danach energetische
    Mittelung (siehe align_spl)."""
    aligned = align_spl(values_db, freqs_hz, align_band_hz)
    return energetic_mean_db(aligned, axis=0)


@dataclass
class SpreadMetrics:
    std_db: np.ndarray
    min_db: np.ndarray
    max_db: np.ndarray
    range_db: np.ndarray
    n_valid: np.ndarray


def spread_metrics(values_db: np.ndarray) -> SpreadMetrics:
    """Streuungsmasse je Frequenzbin ueber alle Positionen (Zeilen), NaN-sicher."""
    values_db = _validate_matrix(values_db)
    n_valid = np.sum(~np.isnan(values_db), axis=0)
    # Die Stichproben-Standardabweichung ist erst ab zwei gueltigen Werten
    # definiert. Solche Bins bleiben NaN; nanstd wird nur darauf angewendet,
    # damit NumPy keine Degrees-of-freedom-RuntimeWarning ausgibt.
    std_db = np.full(values_db.shape[1], np.nan, dtype=np.float64)
    has_sample_std = n_valid >= 2
    if np.any(has_sample_std):
        std_db[has_sample_std] = np.nanstd(
            values_db[:, has_sample_std], axis=0, ddof=1
        )
    with np.errstate(invalid="ignore"):
        min_db = np.nanmin(values_db, axis=0)
        max_db = np.nanmax(values_db, axis=0)
    # nanstd mit ddof=1 liefert NaN, wenn pro Bin nur 1 gueltiger Wert vorliegt
    # (korrekt: Stichprobenstandardabweichung ist dort nicht definiert).
    range_db = max_db - min_db
    return SpreadMetrics(
        std_db=std_db, min_db=min_db, max_db=max_db, range_db=range_db, n_valid=n_valid
    )


@dataclass
class BootstrapBand:
    lower_db: np.ndarray
    upper_db: np.ndarray
    alpha: float
    n_resamples: int


def case_bootstrap_band(
    values_db: np.ndarray,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    rng: np.random.Generator | None = None,
) -> BootstrapBand:
    """Case-Bootstrap (Efron 1979) auf der energetisch gemittelten Kurve.

    Zieht `n_resamples` Mal N Positionsindizes mit Zuruecklegen (N = Anzahl
    Positionen), bildet je Ziehung den energetischen Mittelwert, und gibt
    die (alpha/2, 1-alpha/2)-Perzentile je Frequenzbin zurueck (Default
    alpha=0.10 -> 90%-Band).

    Keine SVD (siehe Modul-Docstring). Bei raeumlich korrelierten,
    benachbarten Positionen (wie hier: Sofamitte + 10 Wanderpunkte um
    dieselbe Sofamitte) unterschaetzt ein naiver Case-Bootstrap die
    tatsaechliche Unsicherheit moeglicherweise - das ist eine bekannte
    Einschraenkung (siehe Zwischenstand-Doku Abschnitt 7a), nicht hier
    geloest.
    """
    values_db = _validate_matrix(values_db)
    n_positions = values_db.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    resampled_means = np.empty((n_resamples, values_db.shape[1]), dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n_positions, size=n_positions)
        resampled_means[i, :] = energetic_mean_db(values_db[idx, :], axis=0)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    with np.errstate(invalid="ignore"):
        lower = np.nanpercentile(resampled_means, lower_pct, axis=0)
        upper = np.nanpercentile(resampled_means, upper_pct, axis=0)
    return BootstrapBand(lower_db=lower, upper_db=upper, alpha=alpha, n_resamples=n_resamples)
