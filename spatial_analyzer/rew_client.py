"""Schreibgeschuetzter REW-API-Client (GET only).

Spricht die lokale REW-API (Standard: http://localhost:4735) an. Verwendet
ausschliesslich HTTP GET. Es gibt in diesem Modul bewusst KEINE Methode fuer
POST/PUT/DELETE oder fuer REW-Prozesskommandos (z. B. Sweep-Start, Cross-Corr-
Align-Trigger) - siehe docs/SPECIFICATION.md Abschnitt 2 ("REW-API-Sicherheit")
und die Zwischenstand-Doku Abschnitt 6a/10.6 zur bewussten Abgrenzung von
Automatisierungsskripten, die REW-Prozesskommandos auslösen.

Messungen werden ausschliesslich ueber ihre stabile UUID angesprochen, nie
ueber einen veraenderlichen numerischen Index.
"""

from __future__ import annotations

import base64
import binascii
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests

DEFAULT_BASE_URL = "http://localhost:4735"
_ALLOWED_METHOD = "GET"


class RewApiError(RuntimeError):
    """Fehler bei der Kommunikation mit der REW-API."""


@dataclass(frozen=True)
class RewMeasurementMeta:
    uuid: str
    title: str
    is_sweep: bool
    raw: dict[str, Any]


class ReadOnlyRewClient:
    """GET-only-Client fuer die REW-API.

    Absichtlich keine Methoden fuer schreibende/prozessauslösende Aufrufe.
    `_request` ist die einzige Stelle, die tatsaechlich Netzwerk-IO macht,
    und ist hart auf HTTP GET beschraenkt.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout_s: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._session = requests.Session()

    def _request(self, path: str, *, raw: bool = False) -> Any:
        url = f"{self._base_url}{path}"
        try:
            response = self._session.request(
                method=_ALLOWED_METHOD, url=url, timeout=self._timeout_s
            )
        except requests.RequestException as exc:  # pragma: no cover - Netzwerkfehler
            raise RewApiError(f"GET {url} fehlgeschlagen: {exc}") from exc
        if response.status_code != 200:
            raise RewApiError(f"GET {url} -> HTTP {response.status_code}")
        if raw:
            return response.content
        try:
            return response.json()
        except ValueError as exc:
            raise RewApiError(f"GET {url} lieferte kein gueltiges JSON") from exc

    def list_measurements(self) -> list[RewMeasurementMeta]:
        data = self._request("/measurements")
        items = self._normalize_measurement_items(data)
        result = []
        for item in items:
            uuid = item.get("uuid") or item.get("id")
            if not uuid:
                continue
            result.append(
                RewMeasurementMeta(
                    uuid=str(uuid),
                    title=item.get("title", ""),
                    is_sweep=bool(item.get("hasImpulseResponse", False)),
                    raw=item,
                )
            )
        return result

    @staticmethod
    def _normalize_measurement_items(data: Any) -> list[dict[str, Any]]:
        """Vereinheitlicht die `/measurements`-Antwort auf eine Liste von Dicts.

        Unterstuetzt: Liste, `{"measurements": [...]}` und das von REW 5.40
        beta 112 beobachtete Dictionary-Format `{"1": {...}, "2": {...}}`.
        Der Dictionary-Schluessel ist ein veraenderlicher Index und wird nie
        als UUID verwendet; Eintraege ohne `uuid`/`id` werden uebersprungen.
        """
        if isinstance(data, dict) and "measurements" in data:
            data = data["measurements"]
        if isinstance(data, dict):
            data = list(data.values())
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def get_measurement_meta(self, uuid: str) -> dict[str, Any]:
        self._require_uuid(uuid)
        return self._request(f"/measurements/{uuid}")

    def get_frequency_response(self, uuid: str) -> dict[str, Any]:
        """Ungeglaetteter Frequenzgang einer Messung per UUID, dekodiert.

        Format am echten REW V5.20.13 belegt (28.11.2026): JSON mit `unit`,
        `smoothing`, `startFreq` und entweder `ppo` (log. Raster) oder
        `freqStep` (lin. Raster); `magnitude`/`phase` sind Base64-kodierte
        Big-Endian-Float32. Ohne Query-Parameter liefert REW die in der GUI
        eingestellte Glaettung (z. B. `Psy`) - daher immer `smoothing=None`.

        Rueckgabe: `freq_hz`, `spl_db`, `phase_deg` (oder None), `smoothing`,
        `unit`.
        """
        self._require_uuid(uuid)
        data = self._request(f"/measurements/{uuid}/frequency-response?smoothing=None")
        return decode_frequency_response(data)

    def get_impulse_response(self, uuid: str) -> dict[str, Any]:
        """Impulsantwort einer Messung per UUID, dekodiert.

        Format am echten REW V5.20.13 belegt (28.11.2026): JSON mit `unit`,
        `startTime`, `sampleInterval`, `sampleRate`, `timingReference` und
        Base64-kodierten Big-Endian-Float32-Samples in `data` (keine rohen
        Bytes).

        Rueckgabe: `samples`, `sample_interval_s`, `start_time_s`,
        `sample_rate_hz`, `timing_reference`, `unit`.
        """
        self._require_uuid(uuid)
        data = self._request(f"/measurements/{uuid}/impulse-response")
        return decode_impulse_response(data)

    def get_impulse_response_raw(self, uuid: str) -> np.ndarray:
        """Nur die Samples der Impulsantwort (siehe `get_impulse_response`)."""
        return self.get_impulse_response(uuid)["samples"]

    @staticmethod
    def _require_uuid(uuid: str) -> None:
        if not uuid or not isinstance(uuid, str):
            raise ValueError(
                "Messungen muessen ueber eine nichtleere UUID adressiert werden, "
                "niemals ueber einen numerischen Index."
            )


def decode_base64_float32(text: Any, field_name: str) -> np.ndarray:
    """Dekodiert ein Base64-Feld der REW-API zu Big-Endian-Float32-Werten."""
    if not isinstance(text, str) or not text:
        raise RewApiError(f"REW-Feld {field_name!r} fehlt oder ist leer")
    try:
        raw_bytes = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RewApiError(f"REW-Feld {field_name!r} ist kein gueltiges Base64") from exc
    return decode_big_endian_float32(raw_bytes)


def _require_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RewApiError(f"REW-Feld {key!r} fehlt oder ist keine endliche Zahl")
    return float(value)


def decode_frequency_response(data: Any) -> dict[str, Any]:
    """Dekodiert die `/frequency-response`-Antwort (Format siehe Client)."""
    if not isinstance(data, dict):
        raise RewApiError("Frequenzgang-Antwort ist kein JSON-Objekt")
    spl = decode_base64_float32(data.get("magnitude"), "magnitude")
    phase = None
    if data.get("phase") is not None:
        phase = decode_base64_float32(data.get("phase"), "phase")
        if len(phase) != len(spl):
            raise RewApiError(
                f"phase ({len(phase)}) und magnitude ({len(spl)}) sind unterschiedlich lang"
            )
    start = _require_number(data, "startFreq")
    if start <= 0:
        raise RewApiError("startFreq muss positiv sein")
    index = np.arange(len(spl), dtype=np.float64)
    if data.get("ppo") is not None:
        ppo = _require_number(data, "ppo")
        if ppo <= 0:
            raise RewApiError("ppo muss positiv sein")
        freq = start * np.power(2.0, index / ppo)
    elif data.get("freqStep") is not None:
        step = _require_number(data, "freqStep")
        if step <= 0:
            raise RewApiError("freqStep muss positiv sein")
        freq = start + index * step
    else:
        raise RewApiError("Frequenzgang-Antwort ohne ppo und ohne freqStep")
    return {
        "freq_hz": freq,
        "spl_db": spl,
        "phase_deg": phase,
        "smoothing": data.get("smoothing"),
        "unit": data.get("unit"),
    }


def decode_impulse_response(data: Any) -> dict[str, Any]:
    """Dekodiert die `/impulse-response`-Antwort (Format siehe Client)."""
    if not isinstance(data, dict):
        raise RewApiError("Impulsantwort-Antwort ist kein JSON-Objekt")
    samples = decode_base64_float32(data.get("data"), "data")
    interval = _require_number(data, "sampleInterval")
    if interval <= 0:
        raise RewApiError("sampleInterval muss positiv sein")
    sample_rate = _require_number(data, "sampleRate") if data.get("sampleRate") is not None else None
    if sample_rate is not None and sample_rate <= 0:
        raise RewApiError("sampleRate muss positiv sein")
    return {
        "samples": samples,
        "sample_interval_s": interval,
        "start_time_s": _require_number(data, "startTime"),
        "sample_rate_hz": sample_rate,
        "timing_reference": data.get("timingReference"),
        "unit": data.get("unit"),
    }


def decode_big_endian_float32(raw_bytes: bytes) -> np.ndarray:
    """Dekodiert einen Big-Endian-Float32-Byte-Stream (REW-API-Binaerformat).

    Wirft RewApiError, wenn die Laenge kein Vielfaches von 4 Bytes ist.
    """
    if len(raw_bytes) % 4 != 0:
        raise RewApiError(
            f"Byte-Stream-Laenge {len(raw_bytes)} ist kein Vielfaches von 4 "
            "(kein gueltiger Float32-Stream)"
        )
    return np.frombuffer(raw_bytes, dtype=">f4").astype(np.float64)
