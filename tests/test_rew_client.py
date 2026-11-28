import base64
import struct

import numpy as np
import pytest

from spatial_analyzer.rew_client import (
    ReadOnlyRewClient,
    RewApiError,
    decode_big_endian_float32,
    decode_frequency_response,
    decode_impulse_response,
)


def _b64(*values: float) -> str:
    return base64.b64encode(struct.pack(f">{len(values)}f", *values)).decode("ascii")


class TestDecodeBigEndianFloat32:
    def test_roundtrip(self):
        values = np.array([1.0, -2.5, 3.14159], dtype=np.float32)
        raw = struct.pack(">3f", *values)
        decoded = decode_big_endian_float32(raw)
        np.testing.assert_allclose(decoded, values, rtol=1e-6)

    def test_rejects_invalid_length(self):
        with pytest.raises(RewApiError):
            decode_big_endian_float32(b"\x00\x01\x02")

    def test_empty_bytes_gives_empty_array(self):
        result = decode_big_endian_float32(b"")
        assert result.size == 0


class TestReadOnlyRewClientGetOnly:
    def test_no_write_methods_exist(self):
        # Diese Klasse darf keine Methoden anbieten, die auf POST/PUT/DELETE
        # oder REW-Prozesskommandos hindeuten (siehe docs/SPECIFICATION.md
        # Abschnitt 2, "REW-API-Sicherheit").
        forbidden_substrings = ("post", "put", "delete", "trigger", "run_sweep", "start_sweep")
        client = ReadOnlyRewClient()
        public_methods = [m for m in dir(client) if not m.startswith("_")]
        for method_name in public_methods:
            lowered = method_name.lower()
            for forbidden in forbidden_substrings:
                assert forbidden not in lowered, (
                    f"Methode {method_name} enthaelt verbotenes Schluesselwort "
                    f"'{forbidden}' - Client muss strikt GET-only bleiben."
                )

    def test_require_uuid_rejects_empty_string(self):
        with pytest.raises(ValueError):
            ReadOnlyRewClient()._require_uuid("")

    def test_require_uuid_accepts_nonempty_string(self):
        # sollte nicht werfen
        ReadOnlyRewClient()._require_uuid("abc-123")

    def test_request_method_is_hardcoded_to_get(self):
        import inspect

        source = inspect.getsource(ReadOnlyRewClient._request)
        assert "_ALLOWED_METHOD" in source
        from spatial_analyzer.rew_client import _ALLOWED_METHOD

        assert _ALLOWED_METHOD == "GET"


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b""):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content

    def json(self):
        return self._json_data


class TestReadOnlyRewClientWithMockedTransport:
    """Testet die fachliche Logik (UUID-Pflicht, JSON-Parsing, Fehlerpfade)
    ohne eine echte REW-Instanz zu benoetigen."""

    def test_list_measurements_parses_uuid_and_sweep_flag(self, monkeypatch):
        client = ReadOnlyRewClient()
        fake_data = [
            {"uuid": "u1", "title": "Sofa L", "hasImpulseResponse": True},
            {"id": "u2", "title": "Sofa R", "hasImpulseResponse": False},
            {"title": "ohne id"},
        ]

        def fake_request(path, raw=False):
            assert path == "/measurements"
            return fake_data

        monkeypatch.setattr(client, "_request", fake_request)
        result = client.list_measurements()
        assert [m.uuid for m in result] == ["u1", "u2"]
        assert result[0].is_sweep is True
        assert result[1].is_sweep is False

    def test_list_measurements_supports_rew_540_dict_format(self, monkeypatch):
        client = ReadOnlyRewClient()
        fake_data = {
            "1": {"uuid": "u1", "title": "Sofa L", "hasImpulseResponse": True},
            "2": {"uuid": "u2", "title": "Sofa R", "hasImpulseResponse": False},
            "3": {"title": "ohne uuid"},
        }
        monkeypatch.setattr(client, "_request", lambda path, raw=False: fake_data)
        result = client.list_measurements()
        # Dict-Schluessel (Index) wird nie als UUID verwendet.
        assert [m.uuid for m in result] == ["u1", "u2"]
        assert result[0].is_sweep is True
        assert result[1].title == "Sofa R"

    def test_list_measurements_supports_wrapped_dict_and_empty(self, monkeypatch):
        client = ReadOnlyRewClient()
        payloads = iter([{"measurements": {"1": {"uuid": "u9"}}}, {}, []])
        monkeypatch.setattr(client, "_request", lambda path, raw=False: next(payloads))
        assert [m.uuid for m in client.list_measurements()] == ["u9"]
        assert client.list_measurements() == []
        assert client.list_measurements() == []

    def test_get_measurement_meta_requires_uuid(self):
        client = ReadOnlyRewClient()
        with pytest.raises(ValueError):
            client.get_measurement_meta("")

    def test_get_frequency_response_requests_unsmoothed_and_decodes(self, monkeypatch):
        # Struktur wie am echten REW V5.20.13 beobachtet (28.11.2026), Werte gekuerzt.
        client = ReadOnlyRewClient()
        seen_paths = []
        payload = {
            "unit": "SPL",
            "smoothing": "None",
            "startFreq": 20.0,
            "freqStep": 0.5,
            "magnitude": _b64(53.81, 53.89, 53.97),
            "phase": _b64(134.9, 137.2, 139.7),
        }

        def fake_request(path, raw=False):
            seen_paths.append(path)
            return payload

        monkeypatch.setattr(client, "_request", fake_request)
        result = client.get_frequency_response("uuid-xyz")
        assert seen_paths == ["/measurements/uuid-xyz/frequency-response?smoothing=None"]
        np.testing.assert_allclose(result["freq_hz"], [20.0, 20.5, 21.0])
        np.testing.assert_allclose(result["spl_db"], [53.81, 53.89, 53.97], rtol=1e-6)
        np.testing.assert_allclose(result["phase_deg"], [134.9, 137.2, 139.7], rtol=1e-6)
        assert result["smoothing"] == "None"

    def test_decode_frequency_response_log_spacing(self):
        result = decode_frequency_response(
            {"startFreq": 20.0, "ppo": 2, "magnitude": _b64(1.0, 2.0, 3.0)}
        )
        np.testing.assert_allclose(result["freq_hz"], [20.0, 20.0 * 2**0.5, 40.0])
        assert result["phase_deg"] is None

    @pytest.mark.parametrize(
        "payload",
        [
            {"startFreq": 20.0, "magnitude": "AAAA"},  # weder ppo noch freqStep
            {"startFreq": 20.0, "ppo": 96},  # magnitude fehlt
            {"startFreq": 20.0, "ppo": 96, "magnitude": "nicht base64!"},
            {"startFreq": 20.0, "ppo": 96, "magnitude": _b64(1.0, 2.0), "phase": _b64(1.0)},
            {"startFreq": 0.0, "ppo": 96, "magnitude": _b64(1.0)},
            {"ppo": 96, "magnitude": _b64(1.0)},
            [],
        ],
    )
    def test_decode_frequency_response_rejects_invalid(self, payload):
        with pytest.raises(RewApiError):
            decode_frequency_response(payload)

    def test_get_impulse_response_decodes_json_base64(self, monkeypatch):
        # REW liefert die IR als JSON mit Base64-Feld `data`, nicht als rohe Bytes.
        client = ReadOnlyRewClient()
        payload = {
            "unit": "percent",
            "startTime": -0.01,
            "sampleInterval": 1 / 44100,
            "sampleRate": 44100.0,
            "timingReference": "Keine Timing-Referenz",
            "data": _b64(1.0, -2.0),
        }

        def fake_request(path, raw=False):
            assert raw is False
            assert path == "/measurements/uuid-xyz/impulse-response"
            return payload

        monkeypatch.setattr(client, "_request", fake_request)
        result = client.get_impulse_response("uuid-xyz")
        np.testing.assert_allclose(result["samples"], [1.0, -2.0])
        assert result["sample_interval_s"] == pytest.approx(1 / 44100)
        assert result["start_time_s"] == -0.01
        assert result["sample_rate_hz"] == 44100.0
        assert result["timing_reference"] == "Keine Timing-Referenz"
        np.testing.assert_allclose(client.get_impulse_response_raw("uuid-xyz"), [1.0, -2.0])

    def test_decode_impulse_response_rejects_invalid(self):
        with pytest.raises(RewApiError):
            decode_impulse_response({"startTime": 0.0, "sampleInterval": 0.0, "data": _b64(1.0)})
        with pytest.raises(RewApiError):
            decode_impulse_response({"startTime": 0.0, "sampleInterval": 1e-5})

    def test_invalid_json_raises_rew_api_error(self, monkeypatch):
        client = ReadOnlyRewClient()

        class _BadJsonResponse:
            status_code = 200
            content = b"kein json"

            def json(self):
                raise ValueError("kein JSON")

        class _Session:
            def request(self, method, url, timeout):
                return _BadJsonResponse()

        monkeypatch.setattr(client, "_session", _Session())
        with pytest.raises(RewApiError):
            client.get_impulse_response("some-uuid")

    @pytest.mark.parametrize("rate", [True, float("nan"), float("inf"), 0.0, "44100"])
    def test_decode_impulse_response_rejects_invalid_sample_rate(self, rate):
        with pytest.raises(RewApiError):
            decode_impulse_response(
                {"startTime": 0.0, "sampleInterval": 1e-5, "sampleRate": rate, "data": _b64(1.0)}
            )

    def test_non_200_status_raises_rew_api_error(self, monkeypatch):
        import requests

        client = ReadOnlyRewClient()

        class _Session:
            def request(self, method, url, timeout):
                assert method == "GET"
                return _FakeResponse(status_code=404)

        monkeypatch.setattr(client, "_session", _Session())
        with pytest.raises(RewApiError):
            client.get_measurement_meta("some-uuid")

    def test_network_exception_is_wrapped_as_rew_api_error(self, monkeypatch):
        import requests

        client = ReadOnlyRewClient()

        class _Session:
            def request(self, method, url, timeout):
                raise requests.ConnectionError("boom")

        monkeypatch.setattr(client, "_session", _Session())
        with pytest.raises(RewApiError):
            client.get_measurement_meta("some-uuid")
