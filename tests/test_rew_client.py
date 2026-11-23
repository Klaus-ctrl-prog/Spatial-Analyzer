import struct

import numpy as np
import pytest

from spatial_analyzer.rew_client import (
    ReadOnlyRewClient,
    RewApiError,
    decode_big_endian_float32,
)


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

    def test_get_frequency_response_calls_expected_path(self, monkeypatch):
        client = ReadOnlyRewClient()
        seen_paths = []

        def fake_request(path, raw=False):
            seen_paths.append(path)
            return {"freq": [20, 40], "spl": [60, 61]}

        monkeypatch.setattr(client, "_request", fake_request)
        result = client.get_frequency_response("uuid-xyz")
        assert seen_paths == ["/measurements/uuid-xyz/frequency-response"]
        assert result["freq"] == [20, 40]

    def test_get_impulse_response_raw_decodes_bytes(self, monkeypatch):
        client = ReadOnlyRewClient()
        raw_bytes = struct.pack(">2f", 1.0, -2.0)

        def fake_request(path, raw=False):
            assert raw is True
            assert path == "/measurements/uuid-xyz/impulse-response"
            return raw_bytes

        monkeypatch.setattr(client, "_request", fake_request)
        result = client.get_impulse_response_raw("uuid-xyz")
        np.testing.assert_allclose(result, [1.0, -2.0])

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
