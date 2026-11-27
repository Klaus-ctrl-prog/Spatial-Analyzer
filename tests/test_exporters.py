import json

import numpy as np
import pytest

from spatial_analyzer.biquad_preview import PeakingFilter
from spatial_analyzer.exporters import (
    auto_preamp_db,
    build_kefplayer_profile,
    build_rew_generic_eq_text,
    export_kefplayer_json,
    export_rew_generic_eq_txt,
    filter_to_kefplayer_dict,
)


class TestFilterToKefplayerDict:
    def test_peq_includes_gain(self):
        f = PeakingFilter(type="peq", fc=100.0, gain_db=-3.0, q=1.5)
        d = filter_to_kefplayer_dict(f)
        assert d == {"type": "peq", "fc": 100.0, "gain_db": -3.0, "q": 1.5}

    def test_lowpass_omits_gain(self):
        f = PeakingFilter(type="lowpass", fc=80.0, q=0.707)
        d = filter_to_kefplayer_dict(f)
        assert "gain_db" not in d
        assert d["type"] == "lowpass"


class TestAutoPreampDb:
    def test_no_filters_gives_zero_preamp(self):
        assert auto_preamp_db([], []) == 0.0

    def test_positive_peak_gives_negative_preamp(self):
        filters = [PeakingFilter(type="peq", fc=1000.0, gain_db=6.0, q=1.0)]
        preamp = auto_preamp_db(filters, [])
        assert preamp < 0
        # Preamp + Peak sollte <= 0 sein
        assert preamp + 6.0 <= 0.01

    def test_only_cuts_give_zero_preamp(self):
        filters = [PeakingFilter(type="peq", fc=1000.0, gain_db=-6.0, q=1.0)]
        assert auto_preamp_db(filters, []) == 0.0


class TestBuildKefplayerProfile:
    def test_schema_matches_expected_keys(self):
        filters_l = [PeakingFilter(type="peq", fc=60.0, gain_db=-3.0, q=5.0)]
        filters_r = []
        profile = build_kefplayer_profile(filters_l, filters_r, name="test", preamp_db=-1.5)
        assert profile["name"] == "test"
        assert profile["preamp_db"] == -1.5
        assert profile["filters"] == []
        assert set(profile["channels"].keys()) == {"L", "R"}
        assert profile["channels"]["L"]["filters"][0]["fc"] == 60.0
        assert profile["channels"]["R"]["filters"] == []

    def test_auto_preamp_is_string_when_none(self):
        profile = build_kefplayer_profile([], [], preamp_db=None)
        assert profile["preamp_db"] == "auto"


class TestExportKefplayerJson:
    def test_writes_valid_json_file(self, tmp_path):
        path = tmp_path / "profile.json"
        filters_l = [PeakingFilter(type="peq", fc=60.0, gain_db=6.0, q=2.0)]
        result = export_kefplayer_json(path, filters_l, [], preamp_db=None)
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["channels"]["L"]["filters"][0]["gain_db"] == 6.0
        # preamp_db sollte konkret aufgeloest sein (nicht "auto"), da ein Boost vorliegt
        assert isinstance(loaded["preamp_db"], float)
        assert loaded == result


class TestBuildRewGenericEqText:
    def test_pk_line_format(self):
        filters = [PeakingFilter(type="peq", fc=100.0, gain_db=-3.0, q=1.41)]
        text = build_rew_generic_eq_text(filters, preamp_db=-2.0)
        assert "Preamp: -2.00 dB" in text
        assert "Filter 1: ON PK Fc 100.00 Hz Gain -3.00 dB Q 1.410" in text

    def test_lowpass_omits_gain_in_line(self):
        filters = [PeakingFilter(type="lowpass", fc=80.0, q=0.707)]
        text = build_rew_generic_eq_text(filters)
        assert "Gain" not in text.split("\n")[-2]  # letzte Filterzeile ohne Gain

    def test_rejects_unknown_type_gracefully(self):
        f = PeakingFilter(type="peq", fc=100.0)
        object.__setattr__(f, "type", "notch")  # Umgehung der Validierung fuer den Test
        with pytest.raises(ValueError):
            build_rew_generic_eq_text([f])


class TestExportRewGenericEqTxt:
    def test_writes_file(self, tmp_path):
        path = tmp_path / "eq.txt"
        filters = [PeakingFilter(type="peq", fc=60.0, gain_db=-3.0, q=5.0)]
        text = export_rew_generic_eq_txt(path, filters, preamp_db=0.0)
        assert path.exists()
        assert path.read_text(encoding="utf-8") == text


def test_export_rew_generic_eq_txt_auto_preamp_and_explicit_zero(tmp_path):
    from spatial_analyzer.exporters import auto_preamp_db

    filters = [PeakingFilter(type="peq", fc=60.0, gain_db=6.0, q=2.0)]
    automatic = export_rew_generic_eq_txt(tmp_path / "auto.txt", filters)
    assert automatic.startswith(f"Preamp: {auto_preamp_db(filters, []):.2f} dB")
    explicit_zero = export_rew_generic_eq_txt(tmp_path / "zero.txt", filters, preamp_db=0.0)
    assert explicit_zero.startswith("Preamp: 0.00 dB")
