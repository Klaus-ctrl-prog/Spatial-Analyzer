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

import struct
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
        return response.content if raw else response.json()

    def list_measurements(self) -> list[RewMeasurementMeta]:
        data = self._request("/measurements")
        items = data if isinstance(data, list) else data.get("measurements", [])
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

    def get_measurement_meta(self, uuid: str) -> dict[str, Any]:
        self._require_uuid(uuid)
        return self._request(f"/measurements/{uuid}")

    def get_frequency_response(self, uuid: str) -> dict[str, Any]:
        """Frequenzgang (SPL/Phase) einer Messung per UUID.

        Hinweis (verifiziert, siehe Zwischenstand-Doku Abschnitt 2): 96 PPO
        impliziert in REW zwingend mindestens 1/48-Oktav-Glaettung. Dieser
        Client liefert nur die Rohantwort der API weiter - die Einordnung
        als "geglaettet" erfolgt in models.Measurement.__post_init__ anhand
        des uebergebenen ppo-Werts, nicht hier.
        """
        self._require_uuid(uuid)
        return self._request(f"/measurements/{uuid}/frequency-response")

    def get_impulse_response_raw(self, uuid: str) -> np.ndarray:
        """Rohe Impulsantwort als Big-Endian-Float32-Array.

        Der IR-Pfad ist laut Zwischenstand-Doku (Abschnitt 2) der zu
        bevorzugende Weg fuer echte ungeglaettete Rohdaten/Phase.
        """
        self._require_uuid(uuid)
        raw_bytes = self._request(f"/measurements/{uuid}/impulse-response", raw=True)
        return decode_big_endian_float32(raw_bytes)

    @staticmethod
    def _require_uuid(uuid: str) -> None:
        if not uuid or not isinstance(uuid, str):
            raise ValueError(
                "Messungen muessen ueber eine nichtleere UUID adressiert werden, "
                "niemals ueber einen numerischen Index."
            )


def decode_big_endian_float32(raw_bytes: bytes) -> np.ndarray:
    """Dekodiert einen Big-Endian-Float32-Byte-Stream (REW-API-Binaerformat).

    Wirft RewApiError, wenn die Laenge kein Vielfaches von 4 Bytes ist.
    """
    if len(raw_bytes) % 4 != 0:
        raise RewApiError(
            f"Byte-Stream-Laenge {len(raw_bytes)} ist kein Vielfaches von 4 "
            "(kein gueltiger Float32-Stream)"
        )
    count = len(raw_bytes) // 4
    values = struct.unpack(f">{count}f", raw_bytes)
    return np.asarray(values, dtype=np.float64)
