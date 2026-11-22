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
        assert args.dry_run is False

    def test_analyze_parses_dry_run_flag(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "in.json", "out_dir", "--dry-run"])
        assert args.dry_run is True

    def test_compare_parses_arguments(self):
        parser = build_parser()
        args = parser.parse_args(["compare", "before_dir", "after_dir", "out_dir"])
        assert args.command == "compare"
        assert args.before_dir == "before_dir"
        assert args.after_dir == "after_dir"
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

    def test_dry_run_valid_input_returns_zero_and_writes_no_report(self, tmp_path, capsys):
        input_path = tmp_path / "input.json"
        _write_minimal_input(input_path)
        output_dir = tmp_path / "out"

        exit_code = main(["analyze", str(input_path), str(output_dir), "--dry-run"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Keine Fehler" in captured.out or "Keine Beanstandungen" in captured.out
        assert not output_dir.exists()

    def test_dry_run_invalid_input_returns_one(self, tmp_path, capsys):
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps({"positions": []}), encoding="utf-8")

        exit_code = main(["analyze", str(input_path), str(tmp_path / "out"), "--dry-run"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "FEHLER" in captured.out

    def test_analyze_without_dry_run_still_blocks_on_invalid_input(self, tmp_path, capsys):
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps({"positions": []}), encoding="utf-8")

        exit_code = main(["analyze", str(input_path), str(tmp_path / "out")])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Fehler" in captured.out

    def test_compare_end_to_end_returns_zero(self, tmp_path, capsys):
        before_in = tmp_path / "before.json"
        after_in = tmp_path / "after.json"
        _write_minimal_input(before_in)
        _write_minimal_input(after_in)
        before_out = tmp_path / "before_out"
        after_out = tmp_path / "after_out"
        main(["analyze", str(before_in), str(before_out)])
        main(["analyze", str(after_in), str(after_out)])

        compare_out = tmp_path / "compare_out"
        exit_code = main(["compare", str(before_out), str(after_out), str(compare_out)])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Vergleichsbericht geschrieben" in captured.out
        assert (compare_out / "compare_report.json").exists()
