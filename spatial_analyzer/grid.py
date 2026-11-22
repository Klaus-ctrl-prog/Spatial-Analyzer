"""R160-Raster-Interpolation (ISO 3 Vorzugszahlenreihe, log-Abstand).

Siehe Zwischenstand-Doku Abschnitt 2 ("R160 preferred log spacing") und
docs/SPECIFICATION.md Abschnitt 3.B: unterschiedliche Frequenzraster werden
nie ueber Array-Indizes gemittelt, sondern zunaechst auf ein gemeinsames
logarithmisches Raster interpoliert - ohne Extrapolation, NaN-sicher.

R160 bedeutet 160 Stufen pro Dekade mit konstantem Verhaeltnisfaktor
10**(1/160) zwischen aufeinanderfolgenden Werten (log-aequidistant).
"""

from __future__ import annotations

import numpy as np

R160_STEPS_PER_DECADE = 160


def r160_grid(start_hz: float, end_hz: float) -> np.ndarray:
    """Erzeugt ein R160-Frequenzraster (log-aequidistant, 160 Stufen/Dekade)
    im Bereich [start_hz, end_hz] (inklusive Endpunkten, soweit sie exakt
    auf dem Raster liegen - sonst wird auf das naechstliegende Rasterintervall
    begrenzt).
    """
    # Haertung (Prioritaet 5): NaN/Inf muessen VOR dem "<= 0"-Vorzeichen-Check
    # explizit abgefangen werden - `float('nan') <= 0` ist `False`, ein NaN
    # wuerde diesen Check also stillschweigend passieren und erst viele
    # Zeilen spaeter bei `int(np.ceil(nan))` mit einem unklaren rohen
    # ValueError ("cannot convert float NaN to integer") abbrechen.
    if not (np.isfinite(start_hz) and np.isfinite(end_hz)):
        raise ValueError(
            f"start_hz ({start_hz}) und end_hz ({end_hz}) muessen endliche Zahlen sein "
            "(kein NaN/Inf)"
        )
    if start_hz <= 0 or end_hz <= 0:
        raise ValueError("start_hz und end_hz muessen positiv sein")
    if start_hz >= end_hz:
        raise ValueError("start_hz muss < end_hz sein")

    log_start = np.log10(start_hz)
    log_end = np.log10(end_hz)
    step = 1.0 / R160_STEPS_PER_DECADE

    first_index = int(np.ceil(log_start / step))
    last_index = int(np.floor(log_end / step))
    if first_index > last_index:
        raise ValueError(
            f"Bereich [{start_hz}, {end_hz}] Hz enthaelt keinen R160-Gitterpunkt"
        )
    indices = np.arange(first_index, last_index + 1)
    return np.power(10.0, indices * step)


def interpolate_to_grid(
    source_freqs_hz: np.ndarray,
    source_values: np.ndarray,
    target_freqs_hz: np.ndarray,
    valid_start_hz: float | None = None,
    valid_end_hz: float | None = None,
) -> np.ndarray:
    """Interpoliert `source_values` (z. B. SPL in dB) linear im log10(f)-Raum
    auf `target_freqs_hz`.

    Keine Extrapolation: Zielpunkte ausserhalb von
    [source_freqs_hz.min(), source_freqs_hz.max()] oder ausserhalb von
    [valid_start_hz, valid_end_hz] (falls angegeben) werden als NaN markiert,
    nicht geraten.

    `source_freqs_hz`/`source_values` werden strikt geprueft (nicht leer,
    kein NaN/Inf, streng monoton steigend) und loesen bei Verstoss eine
    klare `ValueError` aus - ein fehlerhaftes Quellraster soll nicht
    stillschweigend in die Interpolation einfliessen. `target_freqs_hz`
    wird NICHT auf NaN/Inf geprueft: ein NaN-Zielpunkt erfuellt keine der
    Gueltigkeitsbedingungen (jeder Vergleich mit NaN ist `False`) und wird
    dadurch bereits automatisch als NaN im Ergebnis markiert - konsistent
    mit der "keine Extrapolation"-Regel, kein separater Fehlerfall.
    """
    source_freqs_hz = np.asarray(source_freqs_hz, dtype=np.float64)
    source_values = np.asarray(source_values, dtype=np.float64)
    target_freqs_hz = np.asarray(target_freqs_hz, dtype=np.float64)

    if source_freqs_hz.shape != source_values.shape:
        raise ValueError("source_freqs_hz und source_values muessen gleich lang sein")
    # Haertung (Prioritaet 5): explizit VOR der Monotonie-Pruefung, denn
    # `np.diff([...nan...]) <= 0` ist ueberall `False` (NaN-Vergleiche sind
    # nie wahr) - ein NaN im Quellraster wuerde die Monotonie-Pruefung
    # unbemerkt passieren und anschliessend stillschweigend in np.interp()
    # einfliessen (siehe reproduzierter Fall in tests/test_grid.py).
    if source_freqs_hz.size == 0:
        raise ValueError("source_freqs_hz/source_values duerfen nicht leer sein")
    if not np.all(np.isfinite(source_freqs_hz)):
        raise ValueError("source_freqs_hz darf kein NaN/Inf enthalten")
    if not np.all(np.isfinite(source_values)):
        raise ValueError("source_values darf kein NaN/Inf enthalten")
    if np.any(np.diff(source_freqs_hz) <= 0):
        raise ValueError("source_freqs_hz muss streng monoton steigend sein")

    log_source = np.log10(source_freqs_hz)
    log_target = np.log10(target_freqs_hz)

    # np.interp extrapoliert per Default konstant an den Raendern - das wird
    # hier bewusst unterbunden, indem alles ausserhalb des Quellbereichs
    # (und ausserhalb des gueltigen Bereichs) auf NaN gesetzt wird.
    with np.errstate(invalid="ignore"):
        interpolated = np.interp(log_target, log_source, source_values)

    in_source_range = (target_freqs_hz >= source_freqs_hz[0]) & (
        target_freqs_hz <= source_freqs_hz[-1]
    )
    valid_mask = in_source_range
    if valid_start_hz is not None:
        valid_mask &= target_freqs_hz >= valid_start_hz
    if valid_end_hz is not None:
        valid_mask &= target_freqs_hz <= valid_end_hz

    result = np.where(valid_mask, interpolated, np.nan)
    return result
