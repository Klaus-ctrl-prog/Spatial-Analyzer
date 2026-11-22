"""Frequenzabhaengige Fensterung (FDW) auf der ungeglaetteten Roh-Impulsantwort.

Siehe docs/SPECIFICATION.md Abschnitt 3.D.3 ("Frequenzabhaengige Fensterung
(FDW)") und Zwischenstand-Doku Abschnitt 10.2/10.6 (Haertung Prioritaet 7,
2026-11-22: dieser Verweis fehlte hier bisher, obwohl alle anderen D.x-Module
ihn haben - ergaenzt, keine inhaltliche Aenderung).

Eigene, transparent parametrisierte Implementierung - siehe Zwischenstand-Doku
Abschnitt 10.2/10.6: REW erzwingt bei aktivem FDW eine 1/48-Oktav-Glaettung
und es ist nicht verifiziert, ob die REW-API FDW-Parameter ueberhaupt
offenlegt. Dieses Modul rechnet deshalb direkt auf der ueber die REW-API
bezogenen Roh-Impulsantwort, unabhaengig von REW-internen FDW-Einstellungen.

Kein REW-Code, keine REW-interne Fenster-Formel uebernommen - das genaue
REW-eigene FDW-Verhalten (Fensterform, Flankenverlauf) ist nicht verifiziert
und wird hier nicht nachgebildet. Die Parameter (Zyklenzahl, Uebergangsband)
sind bewusst eigene, konfigurierbare Modellannahmen.

Konzept: oberhalb ~500 Hz wird die Impulsantwort mit einer kurzen,
frequenzabhaengigen Fensterlaenge (n Zyklen der jeweiligen Frequenz)
beschnitten, um den Direktschall von spaeten Wandreflexionen zu trennen.
Unterhalb 200 Hz bleibt das volle Zeitfenster erhalten (Modenanalyse).
Zwischen beiden Baendern wird die Fensterlaenge linear im log(f)-Raum
interpoliert, um Stufenartefakte zu vermeiden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_CYCLES = 3.0
DEFAULT_FULL_WINDOW_BELOW_HZ = 200.0
DEFAULT_SHORT_WINDOW_ABOVE_HZ = 500.0


@dataclass(frozen=True)
class FdwConfig:
    cycles: float = DEFAULT_CYCLES
    full_window_below_hz: float = DEFAULT_FULL_WINDOW_BELOW_HZ
    short_window_above_hz: float = DEFAULT_SHORT_WINDOW_ABOVE_HZ

    def __post_init__(self) -> None:
        if self.cycles <= 0:
            raise ValueError("cycles muss positiv sein")
        if not 0 < self.full_window_below_hz < self.short_window_above_hz:
            raise ValueError(
                "es muss 0 < full_window_below_hz < short_window_above_hz gelten"
            )


def window_length_s(freq_hz: np.ndarray, ir_duration_s: float, config: FdwConfig = FdwConfig()) -> np.ndarray:
    """Fensterlaenge (Sekunden) als Funktion der Frequenz.

    - Unterhalb config.full_window_below_hz: volle IR-Laenge (ir_duration_s).
    - Oberhalb config.short_window_above_hz: config.cycles Zyklen der Frequenz.
    - Dazwischen: lineare Interpolation im log10(f)-Raum (weicher Uebergang,
      keine Sprungstelle).
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    if ir_duration_s <= 0:
        raise ValueError("ir_duration_s muss positiv sein")

    short_window_s = config.cycles / freq_hz

    log_f = np.log10(freq_hz)
    log_low = np.log10(config.full_window_below_hz)
    log_high = np.log10(config.short_window_above_hz)
    t = np.clip((log_f - log_low) / (log_high - log_low), 0.0, 1.0)

    # Bei t=0 (<= full_window_below_hz): volle Laenge. Bei t=1
    # (>= short_window_above_hz): kurze, zyklenbasierte Laenge.
    # Lineare Interpolation der Fensterlaenge selbst (nicht log) ist hier
    # bewusst einfach gehalten; die log-Interpolation steckt in t.
    window = (1.0 - t) * ir_duration_s + t * short_window_s
    return np.minimum(window, ir_duration_s)


def apply_fdw(
    ir_samples: np.ndarray,
    sample_interval_s: float,
    peak_index: int,
    target_freqs_hz: np.ndarray,
    config: FdwConfig = FdwConfig(),
) -> np.ndarray:
    """Wendet FDW an und liefert den komplexen Spektralwert je Zielfrequenz.

    Da die Fensterlaenge pro Frequenz unterschiedlich ist, kann keine
    einzelne FFT verwendet werden: jeder Zielfrequenzwert wird direkt als
    gefensterte diskrete Fourier-Summe (Goertzel-aehnlich) ausgewertet.
    Das Fenster ist kausal ab `peak_index` (Direktschall-Peak) mit einer
    Tukey-Flanke (Hann-Taper an beiden Enden), um spektrale Leckagen an den
    Fensterraendern zu reduzieren.
    """
    ir_samples = np.asarray(ir_samples, dtype=np.float64)
    target_freqs_hz = np.asarray(target_freqs_hz, dtype=np.float64)
    n_samples = ir_samples.size
    ir_duration_s = (n_samples - peak_index) * sample_interval_s
    if ir_duration_s <= 0:
        raise ValueError("peak_index liegt nicht vor dem Ende der Impulsantwort")

    window_lengths_s = window_length_s(target_freqs_hz, ir_duration_s, config)
    n = np.arange(n_samples)
    t_rel = (n - peak_index) * sample_interval_s  # Zeit relativ zum Peak

    result = np.empty(target_freqs_hz.shape, dtype=np.complex128)
    for i, (f, win_len_s) in enumerate(zip(target_freqs_hz, window_lengths_s)):
        win_len_samples = max(int(round(win_len_s / sample_interval_s)), 1)
        start = peak_index
        end = min(peak_index + win_len_samples, n_samples)
        segment = ir_samples[start:end]
        if segment.size == 0:
            result[i] = 0.0
            continue
        taper = np.hanning(2 * max(segment.size, 2))[: segment.size] if segment.size > 1 else np.array([1.0])
        # Hanning-Fenster ist symmetrisch; fuer eine einseitige (kausale)
        # Ausblendung wird nur die abfallende zweite Haelfte genutzt, der
        # Anfang (Direktschall-Peak) bleibt ungedaempft.
        fade = np.ones(segment.size)
        fade_len = max(segment.size // 4, 1)
        if segment.size > 1:
            fade[-fade_len:] = np.hanning(2 * fade_len)[fade_len:]
        windowed = segment * fade
        t_segment = t_rel[start:end]
        result[i] = np.sum(windowed * np.exp(-2j * np.pi * f * t_segment))
    return result
