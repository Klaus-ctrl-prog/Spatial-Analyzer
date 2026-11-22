"""Kern-Datenmodell: Measurement, FrequencyGrid, RoomModel.

Alle Groessen, die nicht direkt aus einer Messung stammen (z. B. theoretische
Raummoden aus den Raummassen), werden als Modellwerte gekennzeichnet
(`FieldMeaning.MODEL`) und duerfen nirgends als Messwert behandelt werden
(siehe docs/SPECIFICATION.md, Abschnitt 2, "Physikalische Ehrlichkeit").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

SPEED_OF_SOUND_M_S = 343.0  # Modellannahme (trockene Luft, ~20 Grad C)


class FieldMeaning(str, Enum):
    """Herkunfts-Kennzeichnung eines Feldwerts.

    Analog zur secs-mdat-Spezifikation (siehe claude/secs_mdat_pruefbericht.md):
    jedes geparste Feld wird danach gekennzeichnet, ob es direkt aus der
    Datei gelesen (CONFIRMED), aus anderen Feldern abgeleitet (INFERRED)
    oder unbekannt/nicht belegbar (UNKNOWN) ist. MODEL ist eine Erweiterung
    fuer rein rechnerische Groessen (z. B. theoretische Raummoden), die nie
    aus einer Messung stammen.
    """

    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    UNKNOWN = "unknown"
    MODEL = "model"


class Channel(str, Enum):
    """Messkanal / Signalpfad."""

    L = "L"
    R = "R"
    LR_SUM = "L+R"
    SUB = "Sub"


@dataclass(frozen=True)
class RoomModel:
    """Raumabmessungen und daraus abgeleitete theoretische Modenkandidaten.

    Alle von hier erzeugten Frequenzen sind FieldMeaning.MODEL - ein
    Quaderraum-Modell ersetzt keine Messung (siehe raum_akustik.md /
    Zwischenstand-Doku Abschnitt 5).
    """

    length_m: float
    width_m: float
    height_m: float
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S

    def __post_init__(self) -> None:
        for name, value in (
            ("length_m", self.length_m),
            ("width_m", self.width_m),
            ("height_m", self.height_m),
        ):
            if value <= 0:
                raise ValueError(f"{name} muss positiv sein, war {value}")

    def axial_mode_frequency(self, dimension_m: float, order: int = 1) -> float:
        """f_n = n * c / (2 * L) fuer eine axiale Mode der Ordnung n."""
        if order < 1:
            raise ValueError("order muss >= 1 sein")
        return order * self.speed_of_sound_m_s / (2.0 * dimension_m)

    def axial_mode_candidates(self, max_order: int = 2) -> dict[str, float]:
        """Theoretische axiale Modenkandidaten (Modell, keine Messwerte).

        Schluessel kodieren Dimension und Ordnung, z. B. "length_1",
        "width_1", "height_1", "length_2", ...
        """
        candidates: dict[str, float] = {}
        dims = {"length": self.length_m, "width": self.width_m, "height": self.height_m}
        for dim_name, dim_value in dims.items():
            for order in range(1, max_order + 1):
                key = f"{dim_name}_{order}"
                candidates[key] = self.axial_mode_frequency(dim_value, order)
        return candidates


@dataclass
class FrequencyGrid:
    """Frequenzraster einer Messung mit gueltigem Bereich.

    `frequencies_hz` muss streng monoton steigend sein. `valid_start_hz`
    und `valid_end_hz` begrenzen den Bereich, in dem die zugehoerigen
    Messwerte als belastbar gelten (z. B. UMIK-1-Kalibrierbereich oder
    REW-interner valid_start/valid_end aus der .mdat-Datei).
    """

    frequencies_hz: np.ndarray
    valid_start_hz: float | None = None
    valid_end_hz: float | None = None

    def __post_init__(self) -> None:
        freqs = np.asarray(self.frequencies_hz, dtype=np.float64)
        if freqs.ndim != 1:
            raise ValueError("frequencies_hz muss eindimensional sein")
        if freqs.size < 2:
            raise ValueError("frequencies_hz braucht mindestens 2 Punkte")
        # Haertung (Prioritaet 5, Haertungsphase 2026-11-22): explizit VOR
        # der Monotonie-Pruefung, weil `np.diff([...nan...]) <= 0` wegen der
        # NaN-Vergleichssemantik ueberall `False` ist - ein NaN im
        # Frequenzraster wuerde diese Pruefung sonst unbemerkt passieren
        # (siehe analoger, bereits in grid.py behobener Fund).
        if not np.all(np.isfinite(freqs)):
            raise ValueError("frequencies_hz darf kein NaN/Inf enthalten")
        if np.any(np.diff(freqs) <= 0):
            raise ValueError("frequencies_hz muss streng monoton steigend sein")
        self.frequencies_hz = freqs
        if self.valid_start_hz is None:
            self.valid_start_hz = float(freqs[0])
        if self.valid_end_hz is None:
            self.valid_end_hz = float(freqs[-1])
        if self.valid_start_hz > self.valid_end_hz:
            raise ValueError("valid_start_hz darf nicht > valid_end_hz sein")

    @property
    def valid_mask(self) -> np.ndarray:
        return (self.frequencies_hz >= self.valid_start_hz) & (
            self.frequencies_hz <= self.valid_end_hz
        )

    def overlap_with(self, other: "FrequencyGrid") -> tuple[float, float] | None:
        """Gemeinsamer gueltiger Ueberlappungsbereich zweier Raster, oder None."""
        start = max(self.valid_start_hz, other.valid_start_hz)
        end = min(self.valid_end_hz, other.valid_end_hz)
        if start >= end:
            return None
        return start, end


@dataclass
class ImpulseResponseData:
    """Impulsantwort-Rohdaten (ungeglaettet), analog secs-mdat ImpulseResponse
    (siehe claude/secs_mdat_pruefbericht.md Abschnitt 2 fuer die Referenz-
    Feldliste: samples, sample_interval_s, start_time_s, measurement_delay_s,
    spl_offset_db, peak_index).

    `start_time_s` ist REWs Zeitbezug der Messung, zu unterscheiden von
    `measurement_delay_s` (relative Verzoegerung innerhalb der Impulsantwort
    selbst). Default `None`, nicht `0.0` - fehlt das Feld, ist das nicht
    dasselbe wie ein Zeitstempel von exakt Null.
    """

    samples: np.ndarray
    sample_interval_s: float
    start_time_s: float | None = None
    measurement_delay_s: float = 0.0
    spl_offset_db: float = 0.0
    peak_index: int | None = None
    meaning: FieldMeaning = FieldMeaning.CONFIRMED

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.float64)
        if self.samples.ndim != 1 or self.samples.size == 0:
            raise ValueError("samples muss ein nichtleeres 1D-Array sein")
        # Haertung (Prioritaet 5): `nan <= 0` ist `False`, ein NaN/Inf
        # sample_interval_s wuerde den reinen "<= 0"-Check also unbemerkt
        # passieren (gleiches Muster wie die NaN-Funde in grid.py/
        # FrequencyGrid - siehe dortige Kommentare).
        if not np.isfinite(self.sample_interval_s):
            raise ValueError("sample_interval_s darf kein NaN/Inf sein")
        if self.sample_interval_s <= 0:
            raise ValueError("sample_interval_s muss positiv sein")
        if self.peak_index is None:
            self.peak_index = int(np.argmax(np.abs(self.samples)))

    @property
    def is_sweep(self) -> bool:
        """Heuristik: eine reine RTA-/Betragsmessung hat i.d.R. keine brauchbare
        Impulsantwort (z. B. nur ein einzelner Sample oder reines Rauschen ohne
        klaren Peak). Siehe mdat.py fuer die eigentliche Sweep-vs-RTA-Erkennung;
        dies ist nur ein struktureller Mindest-Check auf Objektebene.
        """
        return self.samples.size > 1


@dataclass
class Measurement:
    """Eine einzelne REW-Messung (eine Position, ein Kanal).

    `uuid` ist die stabile REW-interne Kennung (niemals ein veraenderlicher
    numerischer Index verwenden, siehe docs/SPECIFICATION.md Abschnitt 2).
    """

    uuid: str
    channel: Channel
    position_id: str
    grid: FrequencyGrid
    spl_db: np.ndarray
    phase_deg: np.ndarray | None = None
    unwrapped_phase_deg: np.ndarray | None = None
    group_delay_s: np.ndarray | None = None
    impulse_response: ImpulseResponseData | None = None
    ppo: float | None = None
    forced_smoothing_applied: bool = False
    rew_version: str | None = None
    notes: str = ""
    unknown_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.spl_db = np.asarray(self.spl_db, dtype=np.float64)
        if self.spl_db.shape != self.grid.frequencies_hz.shape:
            raise ValueError(
                "spl_db muss dieselbe Laenge wie grid.frequencies_hz haben "
                f"({self.spl_db.shape} vs {self.grid.frequencies_hz.shape})"
            )
        for name, arr in (
            ("phase_deg", self.phase_deg),
            ("unwrapped_phase_deg", self.unwrapped_phase_deg),
            ("group_delay_s", self.group_delay_s),
        ):
            if arr is not None:
                arr = np.asarray(arr, dtype=np.float64)
                setattr(self, name, arr)
                if arr.shape != self.grid.frequencies_hz.shape:
                    raise ValueError(f"{name} muss dieselbe Laenge wie frequencies_hz haben")
        # 96-PPO-Regel (verifiziert, siehe Zwischenstand-Doku Abschnitt 2):
        # 96 PPO erzwingt bei REW aus Anti-Aliasing-Gruenden mindestens
        # 1/48-Oktav-Glaettung. Das wird hier als Flag mitgefuehrt, nicht
        # stillschweigend korrigiert.
        if self.ppo is not None and self.ppo >= 96:
            self.forced_smoothing_applied = True

    @property
    def has_usable_impulse_response(self) -> bool:
        return self.impulse_response is not None and self.impulse_response.is_sweep

    def spl_in_band(self, start_hz: float, end_hz: float) -> np.ndarray:
        """SPL-Werte im Band [start_hz, end_hz] (nur innerhalb des gueltigen
        Rasterbereichs; keine Extrapolation)."""
        freqs = self.grid.frequencies_hz
        mask = (freqs >= start_hz) & (freqs <= end_hz) & self.grid.valid_mask
        return self.spl_db[mask]
