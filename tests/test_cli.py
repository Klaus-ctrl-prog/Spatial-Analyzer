import json

import numpy as np
import pytest

from spatial_analyzer.cli import build_parser, main


def _write_minimal_input(path):
    freq = np.linspace(20.0, 2000.0, 100)
    spl = np.full_like(freq, 70.0)
    phase = np.zeros_like(freq)
    entry = {"freq_hz": freq.tolist(), "spl_db": spl.tolist(), "phase_deg": phase.tolist()}
    positions = [
        {"id": "sofa_mitte", "l": entry, "r": entry},
        {"id": "wander_1", "l": entry, "r": entry},
    ]
    path.write_text(json.dumps({"room_dimensions_m": [5.72, 3.72, 2.40], "positions": positions}), encoding="utf-8")


class TestBuildParser:
    def test_analyze_requires_input_and_output(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["analyze"])

    def test_analyze_parses_arguments(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "in.json", "out_dir"])
        assert args.command == "analyze"
        assert args.input == "in.json"
        assert args.output_dir == "out_dir"


class TestMain:
    def test_analyze_end_to_end_returns_zero(self, tmp_path, capsys):
        input_path = tmp_path / "input.json"
        _write_minimal_input(input_path)
        output_dir = tmp_path / "out"

        exit_code = main(["analyze", str(input_path), str(output_dir)])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Report geschrieben" in captured.out
        assert (output_dir / "report.json").exists()

    def test_no_command_raises_system_exit(self):
        with pytest.raises(SystemExit):
            main([])
