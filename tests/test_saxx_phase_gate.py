"""Phase-Gate (Saxx ohne Delay) und Filter-Export-Einbindung in Pipeline/CLI.

Rein synthetische Testvektoren; keine Aussage ueber reales Geraeteverhalten.
"""

import json

import numpy as np
import pytest

from spatial_analyzer.cli import main as cli_main
from spatial_analyzer.filter_export import load_filter_export
from spatial_analyzer.pipeline import run_pipeline
from spatial_analyzer.sub_optimizer import SAXX_PHASE_STEPS_DEG, optimize, optimize_phase_only
from tests.test_pipeline import _synthetic_position, _synthetic_position_with_sub

FREQ = np.linspace(40.0, 140.0, 101)


def _lr_sub(n_pos=3, sub_phase_deg=90.0):
    h_lr = np.ones((n_pos, FREQ.size), dtype=np.complex128)
    h_sub = np.full((n_pos, FREQ.size), np.exp(1j * np.deg2rad(sub_phase_deg)), dtype=np.complex128)
    return h_lr, h_sub


class TestOptimizePhaseOnly:
    @pytest.mark.parametrize("true_phase", [0.0, 90.0, 135.0, 270.0])
    def test_finds_aligning_phase_step(self, true_phase):
        h_lr, h_sub = _lr_sub(sub_phase_deg=true_phase)
        target = np.full(FREQ.size, 20 * np.log10(2.0))  # gleichphasige Summe, Pegel 0 dB
        res = optimize_phase_only(FREQ, h_lr, h_sub, target)
        assert res.best_polarity_deg == true_phase
        assert res.best_gain_db == 0.0
        assert res.best_delay_ms == 0.0
        assert res.best_score == pytest.approx(0.0, abs=1e-9)

    def test_grid_shape_and_default_steps(self):
        h_lr, h_sub = _lr_sub()
        res = optimize_phase_only(FREQ, h_lr, h_sub, np.zeros(FREQ.size))
        assert tuple(res.polarity_grid_deg) == SAXX_PHASE_STEPS_DEG
        assert res.score_grid.shape == (8, res.gain_grid_db.size, 1)

    def test_rejects_single_position_and_nan(self):
        h_lr, h_sub = _lr_sub(n_pos=1)
        with pytest.raises(ValueError):
            optimize_phase_only(FREQ, h_lr, h_sub, np.zeros(FREQ.size))
        h_lr, h_sub = _lr_sub()
        h_sub[0, 50] = np.nan
        with pytest.raises(ValueError):
            optimize_phase_only(FREQ, h_lr, h_sub, np.zeros(FREQ.size))


def test_optimize_rejects_non_polarity_angles():
    """Vorher wurde 90 Grad stillschweigend als 180 Grad gerechnet."""
    h_lr, h_sub = _lr_sub()
    with pytest.raises(ValueError, match="optimize_phase_only"):
        optimize(FREQ, h_lr, h_sub, np.zeros(FREQ.size), polarities_deg=(0.0, 90.0))


def _write(path, positions):
    path.write_text(json.dumps({"room_dimensions_m": [5.7, 3.72, 2.4], "positions": positions}), encoding="utf-8")


class TestPipelineFilterExport:
    def test_with_sub_writes_blocked_saxx_export_and_cli_preview(self, tmp_path, capsys):
        _write(tmp_path / "in.json", [_synthetic_position_with_sub(f"p{i}") for i in range(4)])
        report = run_pipeline(tmp_path / "in.json", tmp_path / "out")

        saxx = load_filter_export(tmp_path / "out" / "filter_export_saxx.json")
        assert saxx["target_device"] == "saxx_ds120"
        assert saxx["approval"]["status"] == "BLOCKED"
        assert saxx["approval"]["hardware_validated"] is False
        assert saxx["crossovers"]["phase_deg"] in SAXX_PHASE_STEPS_DEG
        assert saxx["filters"]["eq_proposals_l"] == [] and saxx["filters"]["eq_proposals_r"] == []
        assert any(n.startswith("Saxx-Vorschlag ohne Delay") for n in report.notes)

        kef = load_filter_export(tmp_path / "out" / "filter_export_kef.json")
        assert kef["run_id"] == saxx["run_id"]
        assert kef["filters"]["subwoofer"] == []
        if report.do_no_harm.is_released and (report.eq_proposals_l or report.eq_proposals_r):
            assert kef["approval"]["status"] in ("RELEASED", "BLOCKED")
        else:
            assert kef["approval"]["status"] == "BLOCKED"

        assert cli_main(["saxx-preview", str(tmp_path / "out" / "filter_export_saxx.json")]) == 0
        assert "NICHT FREIGEGEBEN" in capsys.readouterr().out

    def test_without_sub_writes_only_kef_export(self, tmp_path):
        _write(tmp_path / "in.json", [_synthetic_position(f"p{i}") for i in range(4)])
        run_pipeline(tmp_path / "in.json", tmp_path / "out")
        assert (tmp_path / "out" / "filter_export_kef.json").exists()
        assert not (tmp_path / "out" / "filter_export_saxx.json").exists()

    def test_export_is_deterministic(self, tmp_path):
        _write(tmp_path / "in.json", [_synthetic_position_with_sub(f"p{i}") for i in range(4)])
        run_pipeline(tmp_path / "in.json", tmp_path / "a")
        run_pipeline(tmp_path / "in.json", tmp_path / "b")
        for name in ("filter_export_kef.json", "filter_export_saxx.json"):
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_cli_saxx_preview_reports_invalid_file(tmp_path, capsys):
    bad = tmp_path / "x.json"
    bad.write_text("{}", encoding="utf-8")
    assert cli_main(["saxx-preview", str(bad)]) == 1
    assert "nicht moeglich" in capsys.readouterr().out
