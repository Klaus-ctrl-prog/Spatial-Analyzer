"""Eigener Python-Parser fuer das REW-.mdat-Datenmodell (nach secs-mdat-Spezifikation).

Wichtiger Hinweis zur Ehrlichkeit (siehe docs/SPECIFICATION.md Abschnitt 2,
"Physikalische Ehrlichkeit", und Zwischenstand-Doku Abschnitt 10.6/11):

Dieses Modul implementiert die *Datenmodell*-Struktur von .mdat-Dateien
(Feldnamen, Pflichtfelder, FieldMeaning-Kennzeichnung, AllocationBudget,
Sweep-vs-RTA-Erkennung) wie sie in `claude/secs_mdat_pruefbericht.md` anhand
der SECS-2R-Referenzimplementierung dokumentiert wurde - OHNE deren Code zu
uebernehmen (reiner Nachbau nach Spezifikation, lizenzrechtlich unproblematisch).

Was dieses Modul NICHT tut: Es dekodiert keinen echten REW-`.mdat`-Binaer-
Byte-Stream (Java-Objektserialisierung). Der exakte Byte-Level-Aufbau der
REW-Java-Serialisierung ist in dieser Recherchephase nicht mit Messdaten
verifiziert worden (siehe offener Punkt G.4 der Spezifikation: "Validierung
des .mdat-Parsers mit einer echten, unveraenderten REW-.mdat-Datei ... ab
Maerz 2027"). `parse_bytes()` ist deshalb bewusst nur ein Platzhalter, der
klar dokumentiert, was fehlt, statt einen unverifizierten Byte-Level-Aufbau
zu erfinden. `parse_dict()` ist der aktuell nutzbare, testbare Pfad: er
arbeitet auf einer bereits dekodierten dict-Repraesentation (z. B. aus einem
JSON-/Text-Export oder aus der REW-API) und implementiert die eigentliche
fachliche Logik (Pflichtfelder, Versions-Fallback, AllocationBudget,
Sweep-Erkennung).

Architektur-Hinweis (Stand 2026-11-22, im Rahmen der Haertungsarbeit vor
Fruehjahr 2027 geprueft): `MdatRecord`/`MdatParser` sind aktuell NICHT mit
`pipeline.py` verdrahtet - die CLI/Pipeline liest eine eigene, einfachere
JSON-Zwischenform direkt ein (siehe Modul-Docstring von `pipeline.py`), nicht
den Weg ueber dieses Modul. `start_time_s` (REWs Zeitbezug, siehe
`MdatImpulseResponse`) bleibt deshalb nach dem Parsen hier zwar erhalten,
fliesst aber noch in keine nachgelagerte Berechnung ein. Das ist kein neuer
Mangel dieser Haertungsarbeit, sondern ein bestehender, hier nur explizit
benannter Architekturzustand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spatial_analyzer.models import FieldMeaning

# Pflichtfelder auf oberster Ebene, ohne die eine .mdat-Repraesentation nicht
# als valide gelten kann.
REQUIRED_TOP_LEVEL_FIELDS = ("frequency_spacing",)

DEFAULT_MAX_ALLOCATION_BYTES = 256 * 1024 * 1024  # 256 MiB Schutz-Budget


class MdatValidationError(ValueError):
    """Pflichtfeld fehlt oder Struktur ist ungueltig."""


class AllocationBudgetExceeded(MemoryError):
    """Eine korrupte/boesartige .mdat-Struktur hat ein unplausibel grosses
    Array angefordert; Parsing wird abgebrochen statt den Speicher zu fuellen.
    """


@dataclass
class AllocationBudget:
    """Verfolgt kumulative Array-Allokationen waehrend des Parsens.

    Schuetzt gegen korrupte oder praeparierte Dateien, die z. B. eine
    Array-Laenge von mehreren Milliarden angeben.
    """

    max_bytes: int = DEFAULT_MAX_ALLOCATION_BYTES
    _used_bytes: int = field(default=0, repr=False)

    def charge(self, n_elements: int, dtype_size_bytes: int = 8) -> None:
        if n_elements < 0:
            raise MdatValidationError(f"Negative Array-Laenge: {n_elements}")
        requested = n_elements * dtype_size_bytes
        if self._used_bytes + requested > self.max_bytes:
            raise AllocationBudgetExceeded(
                f"Allokation von {requested} Bytes wuerde Budget "
                f"({self.max_bytes} Bytes) ueberschreiten "
                f"(bereits verbraucht: {self._used_bytes})"
            )
        self._used_bytes += requested

    @property
    def used_bytes(self) -> int:
        return self._used_bytes


@dataclass
class MdatFrequencySpacing:
    valid_start_hz: float
    valid_end_hz: float
    ppo: float | None
    meaning: FieldMeaning = FieldMeaning.CONFIRMED


@dataclass
class MdatFrequencyResponse:
    frequency_hz: np.ndarray
    spl_db: np.ndarray
    phase_deg: np.ndarray | None = None
    unwrapped_phase_deg: np.ndarray | None = None
    group_delay_s: np.ndarray | None = None
    meaning: FieldMeaning = FieldMeaning.CONFIRMED

    def __post_init__(self) -> None:
        self.frequency_hz = np.asarray(self.frequency_hz, dtype=np.float64)
        self.spl_db = np.asarray(self.spl_db, dtype=np.float64)
        if self.frequency_hz.shape != self.spl_db.shape:
            raise MdatValidationError("frequency_hz und spl_db muessen gleich lang sein")


@dataclass
class MdatImpulseResponse:
    """Siehe `claude/secs_mdat_pruefbericht.md` Abschnitt 2 ("src/models.rs")
    fuer die Referenz-Feldliste, an der sich dieser Nachbau orientiert:
    `samples, sample_interval_s, start_time_s, measurement_delay_s,
    spl_offset_db, peak_index`.

    `start_time_s` ist REWs absoluter/relativer Zeitbezug einer Messung
    (z. B. Sekunden seit Messstart oder Unix-Zeitstempel, je nach REW-
    Version/Export - hier bewusst nicht weiter interpretiert, nur verlust-
    frei durchgereicht). Er ist von `measurement_delay_s` (relative
    Verzoegerung innerhalb der Impulsantwort selbst, z. B. aus Mikrofon-
    abstand) zu unterscheiden - beide Felder haben unterschiedliche
    Bedeutung und duerfen nicht verwechselt werden.

    Default `None` (nicht `0.0`): fehlt das Feld in der Eingabe, ist das
    nicht dasselbe wie ein Zeitstempel von exakt Null - `None` macht das
    "fehlt" explizit abfragbar, statt einen Messwert vorzutaeuschen.
    """

    samples: np.ndarray
    sample_interval_s: float
    start_time_s: float | None = None
    measurement_delay_s: float = 0.0
    spl_offset_db: float = 0.0
    peak_index: int | None = None
    meaning: FieldMeaning = FieldMeaning.CONFIRMED


@dataclass
class MdatRoomDimensions:
    length_m: float | None
    width_m: float | None
    height_m: float | None
    meaning: FieldMeaning = FieldMeaning.UNKNOWN


@dataclass
class MdatRecord:
    frequency_spacing: MdatFrequencySpacing
    frequency_response: MdatFrequencyResponse | None
    impulse_response: MdatImpulseResponse | None
    room_dimensions: MdatRoomDimensions | None
    notes: str
    rew_version: str
    rew_version_meaning: FieldMeaning
    unknown_fields: dict[str, Any] = field(default_factory=dict)

    @property
    def is_sweep(self) -> bool:
        """Unterscheidung echter Sinus-Sweep (mit nutzbarer Impulsantwort)
        vs. reiner RTA-/Betragsmessung.

        Heuristik (dokumentiert als Modellannahme, kein verifizierter
        REW-interner Schwellenwert): eine Sweep-Messung hat eine
        Impulsantwort mit mehr als einem Sample und einem klar erkennbaren
        Peak, dessen Betrag deutlich (hier: Faktor >= 3) ueber dem
        Median-Betrag der restlichen Samples liegt. Reine RTA-Messungen
        liefern in der Praxis keine brauchbare Impulsantwort.
        """
        ir = self.impulse_response
        if ir is None or ir.samples.size <= 1:
            return False
        abs_samples = np.abs(ir.samples)
        peak_idx = ir.peak_index if ir.peak_index is not None else int(np.argmax(abs_samples))
        peak_value = abs_samples[peak_idx]
        rest = np.delete(abs_samples, peak_idx)
        if rest.size == 0:
            return True
        median_rest = float(np.median(rest))
        if median_rest <= 0:
            return peak_value > 0
        return bool(peak_value >= 3.0 * median_rest)


class MdatParser:
    """Parser fuer die .mdat-Datenmodell-Struktur (Pflichtfeldprueung,
    Versions-Fallback, AllocationBudget-Schutz).

    Arbeitet auf einer bereits dekodierten dict-Repraesentation. Der direkte
    Binaer-Pfad (`parse_bytes`) ist bewusst noch nicht implementiert (siehe
    Modul-Docstring und Spezifikation G.4).
    """

    def __init__(self, max_allocation_bytes: int = DEFAULT_MAX_ALLOCATION_BYTES) -> None:
        self._max_allocation_bytes = max_allocation_bytes

    def parse_dict(
        self, data: dict[str, Any], *, header_rew_version_fallback: str = "unknown"
    ) -> MdatRecord:
        self._check_required_fields(data)
        budget = AllocationBudget(max_bytes=self._max_allocation_bytes)

        freq_spacing = self._parse_frequency_spacing(data["frequency_spacing"])
        freq_response = self._parse_frequency_response(data.get("frequency_response"), budget)
        impulse_response = self._parse_impulse_response(data.get("impulse_response"), budget)
        room_dimensions = self._parse_room_dimensions(data.get("room_dimensions"))

        rew_version = data.get("rew_version")
        if rew_version:
            rew_version_meaning = FieldMeaning.CONFIRMED
        else:
            rew_version = header_rew_version_fallback
            rew_version_meaning = FieldMeaning.INFERRED

        known_keys = {
            "frequency_spacing",
            "frequency_response",
            "impulse_response",
            "room_dimensions",
            "notes",
            "rew_version",
        }
        unknown_fields = {k: v for k, v in data.items() if k not in known_keys}

        if freq_response is None and impulse_response is None:
            raise MdatValidationError(
                "Weder frequency_response noch impulse_response vorhanden - "
                "eine .mdat-Repraesentation ohne jegliche Messdaten ist ungueltig."
            )

        return MdatRecord(
            frequency_spacing=freq_spacing,
            frequency_response=freq_response,
            impulse_response=impulse_response,
            room_dimensions=room_dimensions,
            notes=str(data.get("notes", "")),
            rew_version=str(rew_version),
            rew_version_meaning=rew_version_meaning,
            unknown_fields=unknown_fields,
        )

    def parse_bytes(self, raw: bytes) -> MdatRecord:  # pragma: no cover
        raise NotImplementedError(
            "Byte-Level-Dekodierung der echten REW-.mdat-Java-Serialisierung ist "
            "in dieser Recherchephase nicht verifiziert (keine reale .mdat-Datei "
            "zum Abgleich verfuegbar, siehe docs/SPECIFICATION.md Abschnitt 3.G, "
            "Punkt 4: Validierung erst ab Maerz 2027 moeglich). Bitte stattdessen "
            "parse_dict() mit einer bereits dekodierten Repraesentation verwenden "
            "(z. B. aus einem REW-Text-/JSON-Export oder der REW-API)."
        )

    @staticmethod
    def _check_required_fields(data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise MdatValidationError("Eingabedaten muessen ein dict sein")
        missing = [f for f in REQUIRED_TOP_LEVEL_FIELDS if f not in data]
        if missing:
            raise MdatValidationError(f"Pflichtfeld(er) fehlen: {', '.join(missing)}")

    @staticmethod
    def _parse_frequency_spacing(raw: dict[str, Any]) -> MdatFrequencySpacing:
        try:
            start = float(raw["valid_start_hz"])
            end = float(raw["valid_end_hz"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MdatValidationError(
                "frequency_spacing braucht valid_start_hz und valid_end_hz"
            ) from exc
        if start >= end:
            raise MdatValidationError("valid_start_hz muss < valid_end_hz sein")
        ppo = raw.get("ppo")
        return MdatFrequencySpacing(
            valid_start_hz=start,
            valid_end_hz=end,
            ppo=float(ppo) if ppo is not None else None,
        )

    @staticmethod
    def _parse_frequency_response(
        raw: dict[str, Any] | None, budget: AllocationBudget
    ) -> MdatFrequencyResponse | None:
        if raw is None:
            return None
        try:
            freq = np.asarray(raw["frequency_hz"], dtype=np.float64)
            spl = np.asarray(raw["spl_db"], dtype=np.float64)
        except KeyError as exc:
            raise MdatValidationError(
                "frequency_response braucht frequency_hz und spl_db"
            ) from exc
        budget.charge(freq.size, dtype_size_bytes=8)
        budget.charge(spl.size, dtype_size_bytes=8)
        phase = raw.get("phase_deg")
        unwrapped = raw.get("unwrapped_phase_deg")
        group_delay = raw.get("group_delay_s")
        for arr in (phase, unwrapped, group_delay):
            if arr is not None:
                budget.charge(len(arr), dtype_size_bytes=8)
        return MdatFrequencyResponse(
            frequency_hz=freq,
            spl_db=spl,
            phase_deg=np.asarray(phase, dtype=np.float64) if phase is not None else None,
            unwrapped_phase_deg=(
                np.asarray(unwrapped, dtype=np.float64) if unwrapped is not None else None
            ),
            group_delay_s=(
                np.asarray(group_delay, dtype=np.float64) if group_delay is not None else None
            ),
        )

    @staticmethod
    def _parse_impulse_response(
        raw: dict[str, Any] | None, budget: AllocationBudget
    ) -> MdatImpulseResponse | None:
        if raw is None:
            return None
        try:
            samples = np.asarray(raw["samples"], dtype=np.float64)
            sample_interval_s = float(raw["sample_interval_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MdatValidationError(
                "impulse_response braucht samples und sample_interval_s"
            ) from exc
        budget.charge(samples.size, dtype_size_bytes=8)
        peak_index = raw.get("peak_index")
        start_time_s = raw.get("start_time_s")
        return MdatImpulseResponse(
            samples=samples,
            sample_interval_s=sample_interval_s,
            start_time_s=float(start_time_s) if start_time_s is not None else None,
            measurement_delay_s=float(raw.get("measurement_delay_s", 0.0)),
            spl_offset_db=float(raw.get("spl_offset_db", 0.0)),
            peak_index=int(peak_index) if peak_index is not None else None,
        )

    @staticmethod
    def _parse_room_dimensions(raw: dict[str, Any] | None) -> MdatRoomDimensions | None:
        if raw is None:
            return None
        return MdatRoomDimensions(
            length_m=raw.get("length_m"),
            width_m=raw.get("width_m"),
            height_m=raw.get("height_m"),
            meaning=FieldMeaning.CONFIRMED,
        )
