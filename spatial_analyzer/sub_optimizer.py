"""Subwoofer-Uebergangs- & Delay-Optimierer (KEF LS50 Wireless + Saxx DS 120 DSP).

Siehe docs/SPECIFICATION.md Abschnitt 3.E und Zwischenstand-Doku Abschnitt
6a/11: vektorisierte Grid-Search ueber den vollen 3D-Suchraum
(Pegel g, Delay tau, Polaritaet p) im Uebergangsbereich 50-120 Hz.

H_Gesamt(f, g, tau, p) = H_LR(f) + p * 10**(g/20) * H_Sub(f) * exp(-j*2*pi*f*tau)

Zielfunktion (verbindlich festgelegt, als Modell-Gewichtung gekennzeichnet -
siehe Zwischenstand-Doku Abschnitt 11, Punkt 1):
70% RMSE zur Zielkurve + 30% raeumliche Streuung ueber alle Positionen.
Das bekannte Modenband 60-62 Hz wird getrennt ausgewiesen, damit eine
Verbesserung/Verschlechterung dort nicht als "Mode behoben" statt als
Interferenzeffekt zwischen zwei Quellen fehlinterpretiert wird.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_GAIN_RANGE_DB = (-6.0, 6.0)
DEFAULT_DELAY_RANGE_MS = (0.0, 20.0)
DEFAULT_POLARITIES = (0.0, 180.0)
DEFAULT_MODE_BAND_HZ = (60.0, 62.0)
DEFAULT_TARGET_WEIGHT = 0.7
DEFAULT_SPATIAL_WEIGHT = 0.3


@dataclass(frozen=True)
class ObjectiveWeights:
    """Bewusst gewaehlte Modell-Gewichtung (keine gemessene/abgeleitete
    Zahl) - siehe Zwischenstand-Doku Abschnitt 11, Punkt 1."""

    target_weight: float = DEFAULT_TARGET_WEIGHT
    spatial_weight: float = DEFAULT_SPATIAL_WEIGHT

    def __post_init__(self) -> None:
        total = self.target_weight + self.spatial_weight
        if not np.isclose(total, 1.0):
            raise ValueError(f"target_weight + spatial_weight muss 1.0 ergeben, war {total}")


@dataclass
class SubOptimizerResult:
    best_gain_db: float
    best_delay_ms: float
    best_polarity_deg: float
    best_score: float
    gain_grid_db: np.ndarray
    delay_grid_ms: np.ndarray
    polarity_grid_deg: np.ndarray
    score_grid: np.ndarray
    mode_band_hz: tuple[float, float]
    mode_band_rmse_at_best_db: float


def _complex_sum(
    h_lr: np.ndarray,
    h_sub: np.ndarray,
    gain_db: float,
    delay_ms: float,
    polarity_deg: float,
    freq_hz: np.ndarray,
) -> np.ndarray:
    polarity = 1.0 if np.isclose(polarity_deg % 360.0, 0.0) else -1.0
    gain_lin = 10.0 ** (gain_db / 20.0)
    delay_s = delay_ms / 1000.0
    return h_lr + polarity * gain_lin * h_sub * np.exp(-2j * np.pi * freq_hz * delay_s)


def optimize(
    freq_hz: np.ndarray,
    h_lr_per_position: np.ndarray,
    h_sub_per_position: np.ndarray,
    target_spl_db: np.ndarray,
    band_hz: tuple[float, float] = (50.0, 120.0),
    mode_band_hz: tuple[float, float] = DEFAULT_MODE_BAND_HZ,
    gain_range_db: tuple[float, float] = DEFAULT_GAIN_RANGE_DB,
    gain_step_db: float = 0.5,
    delay_range_ms: tuple[float, float] = DEFAULT_DELAY_RANGE_MS,
    delay_step_ms: float = 0.2,
    polarities_deg: tuple[float, ...] = DEFAULT_POLARITIES,
    weights: ObjectiveWeights = ObjectiveWeights(),
) -> SubOptimizerResult:
    """3D-Grid-Search ueber (g, tau, p) im Uebergangsband.

    `h_lr_per_position` und `h_sub_per_position`: komplexe Uebertragungs-
    funktionen, Form (n_positions, n_freqs) - siehe complex_response.py zur
    Erzeugung aus SPL/Phase. `target_spl_db`: Zielkurve in dB, Form
    (n_freqs,) (siehe docs/SPECIFICATION.md Abschnitt 3.F - als Modell
    gekennzeichnet).
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    h_lr_per_position = np.asarray(h_lr_per_position, dtype=np.complex128)
    h_sub_per_position = np.asarray(h_sub_per_position, dtype=np.complex128)
    target_spl_db = np.asarray(target_spl_db, dtype=np.float64)

    n_positions = h_lr_per_position.shape[0]
    if h_sub_per_position.shape[0] != n_positions:
        raise ValueError("h_lr_per_position und h_sub_per_position brauchen gleich viele Positionen")
    if h_lr_per_position.shape[1] != freq_hz.shape[0] or target_spl_db.shape[0] != freq_hz.shape[0]:
        raise ValueError("freq_hz, h_*_per_position (Spalten) und target_spl_db muessen gleich lang sein")

    band_mask = (freq_hz >= band_hz[0]) & (freq_hz <= band_hz[1])
    if not np.any(band_mask):
        raise ValueError(f"Uebergangsband {band_hz} Hz liegt ausserhalb von freq_hz")
    mode_mask = band_mask & (freq_hz >= mode_band_hz[0]) & (freq_hz <= mode_band_hz[1])

    gains = np.arange(gain_range_db[0], gain_range_db[1] + gain_step_db / 2, gain_step_db)
    delays = np.arange(delay_range_ms[0], delay_range_ms[1] + delay_step_ms / 2, delay_step_ms)

    best_score = np.inf
    best = (0.0, 0.0, polarities_deg[0])
    score_grid = np.empty((len(polarities_deg), len(gains), len(delays)), dtype=np.float64)

    band_freq = freq_hz[band_mask]
    band_target = target_spl_db[band_mask]

    for pi, p in enumerate(polarities_deg):
        for gi, g in enumerate(gains):
            for di, tau in enumerate(delays):
                total_spl_per_position = np.empty((n_positions, band_freq.size), dtype=np.float64)
                for pos in range(n_positions):
                    h_total = _complex_sum(
                        h_lr_per_position[pos, band_mask],
                        h_sub_per_position[pos, band_mask],
                        g,
                        tau,
                        p,
                        band_freq,
                    )
                    with np.errstate(divide="ignore"):
                        total_spl_per_position[pos, :] = 20.0 * np.log10(
                            np.maximum(np.abs(h_total), 1e-12)
                        )
                mean_over_positions = np.mean(total_spl_per_position, axis=0)
                rmse_to_target = float(np.sqrt(np.mean((mean_over_positions - band_target) ** 2)))
                spatial_std = float(np.mean(np.std(total_spl_per_position, axis=0, ddof=1)))
                score = (
                    weights.target_weight * rmse_to_target + weights.spatial_weight * spatial_std
                )
                score_grid[pi, gi, di] = score
                if score < best_score:
                    best_score = score
                    best = (g, tau, p)

    best_gain, best_delay, best_polarity = best
    mode_band_rmse = _mode_band_rmse(
        freq_hz, h_lr_per_position, h_sub_per_position, target_spl_db, mode_mask,
        best_gain, best_delay, best_polarity,
    )

    return SubOptimizerResult(
        best_gain_db=best_gain,
        best_delay_ms=best_delay,
        best_polarity_deg=best_polarity,
        best_score=best_score,
        gain_grid_db=gains,
        delay_grid_ms=delays,
        polarity_grid_deg=np.asarray(polarities_deg, dtype=np.float64),
        score_grid=score_grid,
        mode_band_hz=mode_band_hz,
        mode_band_rmse_at_best_db=mode_band_rmse,
    )


def _mode_band_rmse(
    freq_hz: np.ndarray,
    h_lr_per_position: np.ndarray,
    h_sub_per_position: np.ndarray,
    target_spl_db: np.ndarray,
    mode_mask: np.ndarray,
    gain_db: float,
    delay_ms: float,
    polarity_deg: float,
) -> float:
    if not np.any(mode_mask):
        return float("nan")
    n_positions = h_lr_per_position.shape[0]
    mode_freq = freq_hz[mode_mask]
    mode_target = target_spl_db[mode_mask]
    total_spl = np.empty((n_positions, mode_freq.size), dtype=np.float64)
    for pos in range(n_positions):
        h_total = _complex_sum(
            h_lr_per_position[pos, mode_mask],
            h_sub_per_position[pos, mode_mask],
            gain_db,
            delay_ms,
            polarity_deg,
            mode_freq,
        )
        with np.errstate(divide="ignore"):
            total_spl[pos, :] = 20.0 * np.log10(np.maximum(np.abs(h_total), 1e-12))
    mean_over_positions = np.mean(total_spl, axis=0)
    return float(np.sqrt(np.mean((mean_over_positions - mode_target) ** 2)))
