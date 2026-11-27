import json

import numpy as np
import pytest

from spatial_analyzer.compare import run_compare
from spatial_analyzer.pipeline import run_pipeline


def _synthetic_position(position_id, peak_db=8.0, peak_freq=60.0):
    freq = np.linspace(20.0, 2000.0, 400)
    spl = np.full_like(freq, 70.0)
    spl += peak_db / (1.0 + ((freq - peak_freq) / 3.0) ** 2)
    phase = np.zeros_like(freq)
    entry = {"freq_hz": freq.tolist(), "spl_db": spl.tolist(), "phase_deg": phase.tolist()}
    return {"id": position_id, "l": dict(entry), "r": dict(entry)}


def _write_run(path, peak_db=8.0, overall_level_shift_db=0.0):
    positions = [_synthetic_position("sofa_mitte", peak_db=peak_db)]
    positions += [_synthetic_position(f"wander_{i}", peak_db=peak_db) for i in range(3)]
    if overall_level_shift_db:
        for p in positions:
            for ch in ("l", "r"):
                p[ch]["spl_db"] = [s + overall_level_shift_db for s in p[ch]["spl_db"]]
    data = {"room_dimensions_m": [5.72, 3.72, 2.40], "positions": positions}
    path.write_text(json.dumps(data), encoding="utf-8")


class TestRunCompare:
    def test_produces_expected_output_files(self, tmp_path):
        before_in = tmp_path / "before.json"
        after_in = tmp_path / "after.json"
        _write_run(before_in, peak_db=8.0)
        _write_run(after_in, peak_db=2.0)  # Mode-Cut "wirkt"

        before_out = tmp_path / "before_out"
        after_out = tmp_path / "after_out"
        run_pipeline(before_in, before_out)
        run_pipeline(after_in, after_out)

        compare_out = tmp_path / "compare_out"
        report = run_compare(before_out, after_out, compare_out)

        for name in ("compare_report.json", "compare_report.html", "compare_delta_l.csv", "compare_delta_r.csv"):
            assert (compare_out / name).exists(), f"{name} fehlt"
        assert report.delta_l is not None
        assert report.delta_r is not None

    def test_mode_reduction_shows_up_as_negative_delta_in_mode_band(self, tmp_path):
        before_in = tmp_path / "before.json"
        after_in = tmp_path / "after.json"
        _write_run(before_in, peak_db=8.0)
        _write_run(after_in, peak_db=0.0)  # Peak komplett "weg"

        before_out = tmp_path / "before_out"
        after_out = tmp_path / "after_out"
        run_pipeline(before_in, before_out)
        run_pipeline(after_in, after_out)

        report = run_compare(before_out, after_out, tmp_path / "compare_out")

        # Eng um die Peak-Frequenz (60 Hz) herum, nicht ueber das ganze
        # Modenband gemittelt - der Lorentz-Peak ist schmal (Breite ~3 Hz).
        band_mask = (report.delta_l.freq_hz >= 55.0) & (report.delta_l.freq_hz <= 65.0)
        assert np.nanmean(report.delta_l.delta_db[band_mask]) < -3.0

    def test_identical_runs_yield_near_zero_delta_outside_reference_band_tolerance(self, tmp_path):
        before_in = tmp_path / "before.json"
        after_in = tmp_path / "after.json"
        _write_run(before_in, peak_db=8.0)
        _write_run(after_in, peak_db=8.0)

        before_out = tmp_path / "before_out"
        after_out = tmp_path / "after_out"
        run_pipeline(before_in, before_out)
        run_pipeline(after_in, after_out)

        report = run_compare(before_out, after_out, tmp_path / "compare_out")
        assert abs(np.nanmean(report.delta_l.delta_db)) < 0.5

    def test_alignment_shift_absorbs_overall_level_difference(self, tmp_path):
        before_in = tmp_path / "before.json"
        after_in = tmp_path / "after.json"
        _write_run(before_in, peak_db=8.0, overall_level_shift_db=0.0)
        _write_run(after_in, peak_db=8.0, overall_level_shift_db=6.0)  # nur lauter aufgenommen

        before_out = tmp_path / "before_out"
        after_out = tmp_path / "after_out"
        run_pipeline(before_in, before_out)
        run_pipeline(after_in, after_out)

        report = run_compare(before_out, after_out, tmp_path / "compare_out")
        # Nach Angleichung sollte die reine Pegeldifferenz (6 dB) nicht mehr
        # als generelles Delta durchschlagen - die Kurven sind ja identisch
        # geformt, nur insgesamt lauter.
        assert abs(report.delta_l.alignment_shift_db + 6.0) < 0.5
        assert abs(np.nanmean(report.delta_l.delta_db)) < 0.5

    def test_missing_report_json_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        before_in = tmp_path / "before.json"
        _write_run(before_in)
        before_out = tmp_path / "before_out"
        run_pipeline(before_in, before_out)

        with pytest.raises(FileNotFoundError):
            run_compare(before_out, empty_dir, tmp_path / "compare_out")

    def test_missing_frequency_table_notes_and_skips_channel(self, tmp_path):
        before_in = tmp_path / "before.json"
        after_in = tmp_path / "after.json"
        _write_run(before_in)
        _write_run(after_in)
        before_out = tmp_path / "before_out"
        after_out = tmp_path / "after_out"
        run_pipeline(before_in, before_out)
        run_pipeline(after_in, after_out)
        (after_out / "frequency_table_l.csv").unlink()

        report = run_compare(before_out, after_out, tmp_path / "compare_out")

        assert report.delta_l is None
        assert report.delta_r is not None
        assert any("frequency_table_l.csv" in n for n in report.notes)


def test_custom_alignment_band_and_disabled_alignment(tmp_path):
    from spatial_analyzer.compare import _compute_curve_delta

    freq = np.linspace(15.0, 300.0, 300)
    before = {"freq_hz": freq, "averaged_spl_db": np.full_like(freq, 70.0)}
    after = {"freq_hz": freq, "averaged_spl_db": np.full_like(freq, 76.0)}
    aligned = _compute_curve_delta(before, after, align_band_hz=(20.0, 200.0))
    assert aligned.alignment_shift_db == pytest.approx(-6.0, abs=0.1)
    unaligned = _compute_curve_delta(before, after, align_band_hz=None)
    assert unaligned.alignment_shift_db == 0.0
    assert np.nanmean(unaligned.delta_db) == pytest.approx(6.0, abs=0.1)


def test_run_compare_records_custom_alignment_band(tmp_path):
    before_in = tmp_path / "before.json"
    after_in = tmp_path / "after.json"
    _write_run(before_in, peak_db=8.0)
    _write_run(after_in, peak_db=8.0, overall_level_shift_db=4.0)
    before_out = tmp_path / "before_out"
    after_out = tmp_path / "after_out"
    run_pipeline(before_in, before_out)
    run_pipeline(after_in, after_out)

    report = run_compare(before_out, after_out, tmp_path / "compare_out", align_band_hz=(20.0, 200.0))
    assert report.align_band_hz == (20.0, 200.0)
    data = json.loads((tmp_path / "compare_out" / "compare_report.json").read_text(encoding="utf-8"))
    assert data["align_band_hz"] == [20.0, 200.0]
