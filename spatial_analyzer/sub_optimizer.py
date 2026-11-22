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

Sensitivitaet der 70/30-Gewichtung (Prioritaet 6 der Roadmap, Stand
2026-11-22, siehe tests/test_sub_optimizer.py `TestObjectiveWeightSensitivity`
fuer die konkreten, deterministischen Mehrpunkt-Szenarien): die 70/30-
Standardgewichtung selbst wird durch diese Haertungsrunde NICHT veraendert -
untersucht wurde nur, wie empfindlich das gefundene Optimum (Gain/Delay/
Polaritaet) auf eine Variation der Gewichtung reagiert, bei sonst gleichen
synthetischen Eingaben:

- Stabiles Regime: sind alle Positionen identisch (keine raeumliche
  Streuung), ist das Ergebnis exakt gewichtungsinvariant - der
  Streuungsterm ist ueberall 0 und wird dadurch unabhaengig von
  `spatial_weight` neutral.
- Empfindliches Regime: bei spuerbarer Streuung zwischen den Positionen
  (unterschiedliche Phase und/oder Pegel der Sub-Ankopplung je Position)
  kann bereits eine Verschiebung der Gewichtung zwischen z. B. 80/20 und
  70/30 einen anderen Delay-Kandidaten liefern (haeufiger beobachtet als
  ein Polaritaetswechsel bei den hier untersuchten Szenarien). In einem
  der untersuchten Szenarien liegt die Standardgewichtung 70/30 in der
  Mitte eines breiteren, bis 50/50 stabilen Plateaus; direkt daneben
  (z. B. 95/5 oder 40/60) kippt das Optimum bereits auf einen anderen
  Delay-Kandidaten. Das belegt nur, dass 70/30 in DIESEM synthetischen
  Szenario nicht unmittelbar an einer Kipp-Schwelle liegt - keine
  allgemeine Aussage ueber Stabilitaet bei echten Raum-/Fahrzeugdaten.

Alle Befunde hierzu sind ausschliesslich algorithmische Verifikation an
rein synthetischen, konstruierten Testfaellen - keine Aussage ueber ein
reales Fahrzeug- oder Wohnzimmerverhalten, und kein Beleg dafuer, dass
70/30 (oder eine andere Gewichtung) "richtig" ist.
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
    Zahl) - siehe Zwischenstand-Doku Abschnitt 11, Punkt 1.

    Unveraendert als 70/30 (`DEFAULT_TARGET_WEIGHT`/`DEFAULT_SPATIAL_WEIGHT`)
    belassen. Siehe Modul-Docstring und tests/test_sub_optimizer.py
    `TestObjectiveWeightSensitivity` fuer die Sensitivitaetsanalyse dieser
    Gewichtung gegenueber 50/50 bis 90/10 an synthetischen Szenarien."""

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
    """Phasenrichtige Summe im Frequenzbereich.

    Haertungshinweis (ab 2026-11-22 verifiziert, siehe test_sub_optimizer.py
    `TestSubSampleDelayPrecision`): `delay_s` ist ein kontinuierlicher
    Gleitkommawert, keine Sample-Index-Grosse - die Verzoegerung wird
    ausschliesslich ueber den linearen Phasenterm `exp(-j*2*pi*f*delay_s)`
    auf die bereits im Frequenzbereich vorliegende Uebertragungsfunktion
    angewendet. Es findet an keiner Stelle eine Rundung auf ganze Samples
    statt (kein `int(delay_s / sample_interval_s)`, kein Array-Verschieben)
    - die Aufloesung ist dadurch ausschliesslich durch die Gleitkomma-
    genauigkeit und die Schrittweite des Such-Grids (`delay_step_ms` in
    `optimize()`) begrenzt, nicht durch die Sample-Periode der zugrunde-
    liegenden Messung.
    """
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

    Bekannte, dokumentierte Grenze - 360-Grad-Periodizitaet bei schmalen
    Baendern (verifiziert, siehe test_sub_optimizer.py
    `TestNarrowBandDelayAmbiguity`): der Phasenterm `exp(-j*2*pi*f*tau)` ist
    fuer eine EINZELNE Frequenz f periodisch in tau mit Periode 1/f. Fuer ein
    schmales Band (z. B. `DEFAULT_MODE_BAND_HZ` = 60-62 Hz) sind Delays, die
    sich um ein ganzzahliges Vielfaches von ~1/f_mitte unterscheiden (bei
    61 Hz: ~16.39 ms), deshalb NAHEZU, aber nicht exakt, gleichwertig - die
    Scores an solchen Kandidaten liegen typischerweise nur im Bereich von
    1e-4 (Score-Einheiten) auseinander, waehrend ein eindeutig falscher
    Delay (z. B. die halbe Periode) einen um Groessenordnungen schlechteren
    Score ergibt. Diese Mehrdeutigkeit ist eine Eigenschaft der Zielfunktion
    bei schmalen Baendern, kein Implementierungsfehler - `delay_range_ms`
    sollte deshalb nicht pauschal weit ueber mehrere Perioden hinaus
    gewaehlt werden, ohne diese Mehrdeutigkeit beim Interpretieren des
    Ergebnisses zu beruecksichtigen. Alle hierzu gehoerenden Tests sind
    ausschliesslich algorithmische Verifikation an synthetischen Daten,
    keine Aussage ueber ein reales Fahrzeug- oder Wohnzimmerverhalten.
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

    # Haertung (ab 2026-11-22, Prioritaet 2 der Roadmap): np.mean()/np.std()
    # unten propagieren NaN stillschweigend durch die GESAMTE Zielfunktion -
    # ein einziger NaN-Bin (z. B. aus einer Interpolation ausserhalb des
    # Quellbereichs, siehe grid.py) macht score_grid an jedem Gitterpunkt zu
    # NaN. Da `NaN < best_score` immer False ist, wuerde `best` dann NIE
    # aktualisiert und die Funktion liefert unbemerkt den Default
    # (gain=0, delay=0, erste Polaritaet, score=inf) zurueck, als sei das
    # ein echtes Optimierungsergebnis. Deshalb hier hart und fruehzeitig
    # pruefen, statt das stillschweigend durchzureichen.
    for name, arr in (
        ("h_lr_per_position", h_lr_per_position[:, band_mask]),
        ("h_sub_per_position", h_sub_per_position[:, band_mask]),
        ("target_spl_db", target_spl_db[band_mask]),
        ("freq_hz", freq_hz[band_mask]),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"{name} enthaelt NaN/Inf innerhalb des Uebergangsbands {band_hz} Hz - "
                "die Grid-Search wuerde sonst stillschweigend einen bedeutungslosen "
                "Default (gain=0, delay=0, score=inf) statt eines echten Ergebnisses "
                "liefern. Bitte vorher interpolieren/maskieren (siehe grid.py) oder "
                "das Band so waehlen, dass es den betroffenen Bereich ausschliesst."
            )

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
