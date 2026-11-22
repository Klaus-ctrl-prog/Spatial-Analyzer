import json

import numpy as np
import pytest

from spatial_analyzer.biquad_preview import PeakingFilter
from spatial_analyzer.eq_solver import EqProposal, ModeConfirmation
from spatial_analyzer.reporting import Report, write_frequency_table_csv
from spatial_analyzer.sub_optimizer import SubOptimizerResult


def _sample_report():
    return Report(
        generated_at="2026-11-22T00:00:00+00:00",
        room_dimensions_m=(5.72, 3.72, 2.40),
        measurement_summary={"reference_level_db": 70.0},
        mode_confirmations=[
            ModeConfirmation(freq_hz=60.0, classification="mode_candidate", decay_confirmed=True, q_factor=5.0)
        ],
        sub_optimizer_result=SubOptimizerResult(
            best_gain_db=2.0, best_delay_ms=3.5, best_polarity_deg=0.0, best_score=0.42,
            gain_grid_db=np.array([0.0]), delay_grid_ms=np.array([0.0]),
            polarity_grid_deg=np.array([0.0]), score_grid=np.zeros((1, 1, 1)),
            mode_band_hz=(60.0, 62.0), mode_band_rmse_at_best_db=1.2,
        ),
        eq_proposals_l=[EqProposal(filter=PeakingFilter(type="peq", fc=60.0, gain_db=-3.0, q=5.0), reason="test")],
        eq_proposals_r=[],
        notes=["Testhinweis"],
    )


class TestReportToDict:
    def test_contains_separated_sections(self):
        report = _sample_report()
        d = report.to_dict()
        assert "messwerte" in d
        assert "modellannahmen" in d
        assert "empfehlungen" in d
        assert d["modellannahmen"]["mode_confirmations"][0]["freq_hz"] == 60.0
        assert d["empfehlungen"]["eq_proposals_l"][0]["gain_db"] == -3.0

    def test_none_sub_optimizer_result_handled(self):
        report = _sample_report()
        report.sub_optimizer_result = None
        d = report.to_dict()
        assert d["modellannahmen"]["sub_optimizer"] is None


class TestReportWriteJson:
    def test_writes_valid_json(self, tmp_path):
        report = _sample_report()
        path = tmp_path / "report.json"
        report.write_json(path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["room_dimensions_m"] == [5.72, 3.72, 2.40]


class TestReportWriteHtml:
    def test_writes_html_with_expected_sections(self, tmp_path):
        report = _sample_report()
        path = tmp_path / "report.html"
        report.write_html(path)
        text = path.read_text(encoding="utf-8")
        assert "<html" in text
        assert "Modellannahmen" in text
        assert "60.0 Hz" in text

    def test_escapes_html_in_notes(self, tmp_path):
        report = _sample_report()
        report.notes = ["<script>alert(1)</script>"]
        path = tmp_path / "report.html"
        report.write_html(path)
        text = path.read_text(encoding="utf-8")
        assert "<script>alert(1)</script>" not in text
        assert "&lt;script&gt;" in text


class TestWriteFrequencyTableCsv:
    def test_writes_expected_columns(self, tmp_path):
        path = tmp_path / "table.csv"
        freq = np.array([20.0, 40.0, 60.0])
        write_frequency_table_csv(path, freq, {"spl_db": np.array([60.0, 61.0, 62.0])})
        content = path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert lines[0] == "freq_hz,spl_db"
        assert len(lines) == 4

    def test_handles_nan_values(self, tmp_path):
        path = tmp_path / "table.csv"
        freq = np.array([20.0, 40.0])
        write_frequency_table_csv(path, freq, {"spl_db": np.array([60.0, np.nan])})
        content = path.read_text(encoding="utf-8")
        assert content.strip().split("\n")[-1] == "40.0000,"

    def test_rejects_mismatched_column_length(self, tmp_path):
        path = tmp_path / "table.csv"
        freq = np.array([20.0, 40.0, 60.0])
        with pytest.raises(ValueError):
            write_frequency_table_csv(path, freq, {"spl_db": np.array([60.0, 61.0])})
