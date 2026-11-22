"""RBJ-Biquad-Mehrpunktvorschau (Vorher/Nachher-Simulation).

Siehe docs/SPECIFICATION.md Abschnitt 3.D.5: Simulation vorgeschlagener
PEQ-Filter mit den exakten RBJ-Biquad-Formeln auf allen 11 L-/R-Positionen
sowie kanalgetrennt auf L+R (H_L*L + H_R*R).

Die Filterformeln (peq/lowshelf/highshelf/lowpass/highpass) sind bewusst
identisch zu den in `Klaus-ctrl-prog/KEFplayer` (`kefplayer/dsp.py`)
verwendeten "Audio EQ Cookbook" (RBJ)-Formeln aufgebaut (gleiche Variablen-
benennung, gleiche A/alpha/w0-Normierung, gleiche a0-Normierung der SOS-
Koeffizienten) - eigenstaendig fuer dieses MIT-lizenzierte Projekt
reimplementiert, kein Code-Copy-Paste, aber bewusst dieselbe Mathematik,
damit eine Vorschau hier mit einem spaeter in KEFplayer geladenen Profil
uebereinstimmt. Hinweis aus KEFplayer selbst: ob die Shelf-Filter bei
gleichem Q exakt wie in Equalizer APO oder REW klingen, ist NICHT
verifiziert (unterschiedliche Q-Konventionen moeglich) - dieser Hinweis
gilt hier unveraendert.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

SQRT_HALF = math.sqrt(0.5)
PEAKING_TYPES = ("peq", "lowshelf", "highshelf", "lowpass", "highpass")


@dataclass(frozen=True)
class PeakingFilter:
    """Ein einzelner Biquad-Filterabschnitt (RBJ Audio-EQ-Cookbook)."""

    type: str
    fc: float
    gain_db: float = 0.0
    q: float = SQRT_HALF

    def __post_init__(self) -> None:
        if self.type not in PEAKING_TYPES:
            raise ValueError(f"Unbekannter Filtertyp {self.type!r}. Erlaubt: {PEAKING_TYPES}")
        if self.fc <= 0:
            raise ValueError("fc muss > 0 sein")
        if self.q <= 0:
            raise ValueError("q muss > 0 sein")

    def sos(self, fs: float) -> np.ndarray:
        """SOS-Zeile [b0 b1 b2 1 a1 a2], normiert durch a0 - identisch zur
        in KEFplayer verwendeten RBJ-Normierung."""
        if self.fc >= fs / 2:
            raise ValueError(f"fc {self.fc} Hz liegt bei fs={fs} Hz nicht unter Nyquist")
        w0 = 2.0 * math.pi * self.fc / fs
        cosw, sinw = math.cos(w0), math.sin(w0)
        alpha = sinw / (2.0 * self.q)
        A = 10.0 ** (self.gain_db / 40.0)
        t = self.type
        if t == "peq":
            b0, b1, b2 = 1 + alpha * A, -2 * cosw, 1 - alpha * A
            a0, a1, a2 = 1 + alpha / A, -2 * cosw, 1 - alpha / A
        elif t == "lowpass":
            b0, b1, b2 = (1 - cosw) / 2, 1 - cosw, (1 - cosw) / 2
            a0, a1, a2 = 1 + alpha, -2 * cosw, 1 - alpha
        elif t == "highpass":
            b0, b1, b2 = (1 + cosw) / 2, -(1 + cosw), (1 + cosw) / 2
            a0, a1, a2 = 1 + alpha, -2 * cosw, 1 - alpha
        elif t == "lowshelf":
            s = 2.0 * math.sqrt(A) * alpha
            b0 = A * ((A + 1) - (A - 1) * cosw + s)
            b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
            b2 = A * ((A + 1) - (A - 1) * cosw - s)
            a0 = (A + 1) + (A - 1) * cosw + s
            a1 = -2 * ((A - 1) + (A + 1) * cosw)
            a2 = (A + 1) + (A - 1) * cosw - s
        else:  # highshelf
            s = 2.0 * math.sqrt(A) * alpha
            b0 = A * ((A + 1) + (A - 1) * cosw + s)
            b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
            b2 = A * ((A + 1) + (A - 1) * cosw - s)
            a0 = (A + 1) - (A - 1) * cosw + s
            a1 = 2 * ((A - 1) - (A + 1) * cosw)
            a2 = (A + 1) - (A - 1) * cosw - s
        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def response_complex(filters: list[PeakingFilter], fs: float, freqs_hz: np.ndarray) -> np.ndarray:
    """Komplexer Frequenzgang einer Filterkette, via scipy.signal.sosfreqz
    (wie in KEFplayer dsp.py: Profile.response_db)."""
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    if not filters:
        return np.ones_like(freqs_hz, dtype=np.complex128)
    sos = np.vstack([f.sos(fs) for f in filters])
    _, h = signal.sosfreqz(sos, worN=freqs_hz, fs=fs)
    return h


@dataclass
class MultiPositionPreview:
    """Vorher/Nachher-SPL (dB) je Position fuer L, R und L+R."""

    freq_hz: np.ndarray
    position_ids: list[str]
    l_before_db: np.ndarray  # (n_positions, n_freqs)
    l_after_db: np.ndarray
    r_before_db: np.ndarray
    r_after_db: np.ndarray
    lr_before_db: np.ndarray
    lr_after_db: np.ndarray


def simulate_multi_position_preview(
    freq_hz: np.ndarray,
    position_ids: list[str],
    l_complex: np.ndarray,
    r_complex: np.ndarray,
    filters_l: list[PeakingFilter],
    filters_r: list[PeakingFilter],
    fs: float,
) -> MultiPositionPreview:
    """Simuliert die Anwendung kanalgetrennter Filterketten auf alle
    uebergebenen Positionen.

    `l_complex`/`r_complex`: komplexe Uebertragungsfunktionen, Form
    (n_positions, n_freqs) - siehe complex_response.to_complex().
    Allgemeiner Fall (siehe Zwischenstand-Doku Abschnitt 10.4/10.6):
    LR_nachher = H_L*L + H_R*R, kanalgetrennt VOR der Summation gefiltert -
    H*(L+R) ist nur der Spezialfall identischer Filter, hier nicht
    vorausgesetzt.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    l_complex = np.asarray(l_complex, dtype=np.complex128)
    r_complex = np.asarray(r_complex, dtype=np.complex128)
    n_positions = l_complex.shape[0]
    if r_complex.shape[0] != n_positions or len(position_ids) != n_positions:
        raise ValueError("l_complex, r_complex und position_ids muessen gleich viele Positionen haben")

    h_l = response_complex(filters_l, fs, freq_hz)
    h_r = response_complex(filters_r, fs, freq_hz)

    l_after = l_complex * h_l[np.newaxis, :]
    r_after = r_complex * h_r[np.newaxis, :]
    lr_before = l_complex + r_complex
    lr_after = l_after + r_after

    def to_db(x: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return 20.0 * np.log10(np.maximum(np.abs(x), 1e-12))

    return MultiPositionPreview(
        freq_hz=freq_hz,
        position_ids=list(position_ids),
        l_before_db=to_db(l_complex),
        l_after_db=to_db(l_after),
        r_before_db=to_db(r_complex),
        r_after_db=to_db(r_after),
        lr_before_db=to_db(lr_before),
        lr_after_db=to_db(lr_after),
    )
