import contextlib
import csv
from dataclasses import fields, replace
import io
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from utils.logger import MeasurementCSV
from utils.measurement_window import MeasurementResult
from utils.reporting import (
    AnalysisReports, METRICS, SUMMARY_COLUMNS, blink_rate, check_outputs, main,
    measurement_status, metric_figure, read_measurements, report_paths,
    summary_values, terminal_table,
)


def result(**changes):
    base = MeasurementResult(
        user_id="test_user", window_index=1,
        start_sec=5.023767, end_sec=65.023767, duration_sec=60,
        blink_count=15, mean_blink_duration_ms=138.669444123,
        duration_sample_count=12, mean_eye_opening_reduction_pct=0.659934123,
        eor_sample_count=1694, face_detection_success_pct=100,
        frame_count=1769, face_detected_count=1769, calibrated=True,
        is_partial=False, measurement_valid=True, end_reason="interval",
    )
    return replace(base, **changes)


def partial_result(**changes):
    base = result(window_index=2, start_sec=65.023767, end_sec=113.743867,
                  duration_sec=48.7201, blink_count=21, mean_blink_duration_ms=193.5,
                  duration_sample_count=21, mean_eye_opening_reduction_pct=1.035182,
                  is_partial=True, end_reason="eof")
    return replace(base, **changes)


class SummaryTests(unittest.TestCase):
    def test_requested_korean_columns_and_display_precision(self):
        values = dict(zip(SUMMARY_COLUMNS, summary_values(result())))
        self.assertEqual(len(values), 8)
        self.assertEqual(values["측정 구간"], "00:05~01:05")
        self.assertEqual(values["평균 눈 감김 시간(ms)"], "138.7")
        self.assertEqual(values["눈 뜸 크기 감소율(%)"], "0.66")
        self.assertEqual(values["분당 깜빡임 환산 횟수(회/분)"], "15.0")
        self.assertEqual(values["측정 상태"], "정상")

    def test_partial_rate_uses_actual_duration_and_keeps_partial_label(self):
        row = partial_result()
        self.assertAlmostEqual(blink_rate(row), 21 * 60 / 48.7201)
        values = dict(zip(SUMMARY_COLUMNS, summary_values(row)))
        self.assertEqual(values["분당 깜빡임 환산 횟수(회/분)"], "25.9")
        self.assertEqual(values["측정 시간(초)"], "48.72")
        self.assertEqual(values["눈 뜸 크기 감소율(%)"], "1.04")
        self.assertEqual(values["측정 상태"], "부분 구간")

    def test_invalid_partial_is_not_relabelled_as_normal_or_only_partial(self):
        self.assertEqual(measurement_status(result(measurement_valid=False)), "유효하지 않음")
        self.assertEqual(measurement_status(partial_result(measurement_valid=False)),
                         "유효하지 않음 (부분 구간)")

    def test_missing_values_are_not_replaced_with_zero(self):
        row = result(mean_blink_duration_ms=None, mean_eye_opening_reduction_pct=None,
                     measurement_valid=False, duration_sec=0)
        values = summary_values(row)
        self.assertEqual(values[3:6], ("", "", ""))
        self.assertIn("N/A", terminal_table([row]))
        self.assertIn("유효하지 않음", terminal_table([row]))

    def test_terminal_table_and_empty_results(self):
        table = terminal_table([result(), partial_result()])
        self.assertIn("측정 구간", table)
        self.assertIn("깜빡임(회/분)", table)
        self.assertIn("25.9", table)
        self.assertIn("138.7", table)
        self.assertIn("부분 구간", table)
        self.assertIn("측정 결과 없음", terminal_table([]))


class ReportFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name) / "user1.csv"

    def test_output_names_accept_base_or_raw_name(self):
        paths = report_paths(self.base)
        self.assertEqual(paths.raw.name, "user1_raw.csv")
        self.assertEqual(paths.summary.name, "user1_summary.csv")
        self.assertEqual(len(paths.graphs), 3)
        self.assertEqual(paths, report_paths(paths.raw))

    def test_raw_is_byte_identical_to_original_writer_and_both_csvs_flush(self):
        original_path = self.base.with_name("original.csv")
        row = result()
        with MeasurementCSV(original_path) as original:
            original.write(row)
        with AnalysisReports(self.base) as reports:
            reports.write(row)
            self.assertEqual(reports.paths.raw.read_bytes(), original_path.read_bytes())
            self.assertEqual(len(next(csv.reader(io.StringIO(
                reports.paths.raw.read_text(encoding="utf-8-sig"))))), 17)
            summary_bytes = reports.paths.summary.read_bytes()
            self.assertTrue(summary_bytes.startswith(b"\xef\xbb\xbf"))
            records = list(csv.DictReader(io.StringIO(summary_bytes.decode("utf-8-sig"))))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["평균 눈 감김 시간(ms)"], "138.7")
            self.assertEqual(row.mean_blink_duration_ms, 138.669444123)

    def test_preflight_checks_every_output_before_creating_raw(self):
        paths = report_paths(self.base)
        for collision in (paths.summary, paths.graphs[0]):
            collision.write_text("keep me", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                AnalysisReports(self.base)
            self.assertFalse(paths.raw.exists())
            self.assertEqual(collision.read_text(encoding="utf-8"), "keep me")
            collision.unlink()

    def test_input_cannot_be_overwritten_even_with_overwrite(self):
        path = report_paths(self.base).raw
        path.write_text("input", encoding="utf-8")
        with self.assertRaises(ValueError):
            check_outputs((path,), source=path, overwrite=True)

    def test_read_preserves_false_flags_and_missing_values(self):
        path = report_paths(self.base).raw
        with MeasurementCSV(path) as writer:
            writer.write(partial_result(measurement_valid=False, mean_blink_duration_ms=None))
        row, = read_measurements(path)
        self.assertFalse(row.measurement_valid)
        self.assertTrue(row.is_partial)
        self.assertIsNone(row.mean_blink_duration_ms)
        self.assertEqual(len(fields(row)), 17)

    def test_malformed_boolean_is_rejected_instead_of_becoming_true(self):
        path = report_paths(self.base).raw
        with MeasurementCSV(path) as writer:
            writer.write(result())
        path.write_text(path.read_text(encoding="utf-8-sig").replace("True", "maybe"),
                        encoding="utf-8-sig")
        with self.assertRaisesRegex(ValueError, "Invalid boolean"):
            read_measurements(path)

    def test_existing_raw_regenerates_summary_and_three_real_pngs(self):
        paths = report_paths(self.base)
        with MeasurementCSV(paths.raw) as writer:
            writer.write(result())
            writer.write(partial_result())
        before = paths.raw.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = main([str(paths.raw)])
        self.assertEqual(status, 0)
        self.assertEqual(paths.raw.read_bytes(), before)
        self.assertIn("25.9", output.getvalue())
        self.assertTrue(paths.summary.exists())
        for path in paths.graphs:
            data = path.read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", data[16:24]), (1500, 720))


class GraphTests(unittest.TestCase):
    def test_invalid_points_are_gaps_not_normal_values_or_fabricated_zeros(self):
        rows = [result(), partial_result(measurement_valid=False)]
        for metric, title, ylabel in METRICS:
            with self.subTest(metric=metric):
                figure = metric_figure(rows, metric, title, ylabel)
                axis = figure.axes[0]
                xs, ys = axis.lines[0].get_data()
                self.assertAlmostEqual(xs[0], (5.023767 + 65.023767) / 120)
                self.assertTrue(math.isfinite(ys[0]))
                self.assertTrue(math.isnan(ys[1]))
                labels = [text.get_text() for text in axis.get_legend().get_texts()]
                self.assertIn("Invalid (excluded)", labels)
                self.assertIn("Partial interval", labels)
                self.assertEqual(len(axis.patches), 2)  # partial band + invalid band
                figure.clear()

    def test_partial_point_is_a_rate_not_raw_count(self):
        figure = metric_figure([partial_result()], *METRICS[0])
        self.assertAlmostEqual(figure.axes[0].lines[0].get_ydata()[0], 21 * 60 / 48.7201)
        figure.clear()

    def test_empty_and_all_invalid_graphs_render_without_normal_data(self):
        for rows in ([], [result(measurement_valid=False, mean_blink_duration_ms=None)]):
            figure = metric_figure(rows, *METRICS[1])
            figure.canvas.draw()
            self.assertTrue(figure.axes[0].texts)
            figure.clear()


class AnalysisEntryPointTests(unittest.TestCase):
    def run_analysis(self, base, *, plot_error=False):
        from analyze_video import main as analyze_main
        from camera.video_file import VideoFrame

        source = Mock(fps=30, expected_frames=2, end_time=65.1,
                      clock=SimpleNamespace(used_fps_fallback=False))
        source.read.side_effect = [VideoFrame(Mock(), 5, 0), VideoFrame(Mock(), 65, 1), None]
        blink = Mock()
        blink.update.side_effect = lambda image, timestamp: {
            "ear": 0.3, "face_detected": True, "blink_event": timestamp == 65,
        }
        duration = Mock()
        duration.update.side_effect = lambda ear, timestamp, face_detected: {
            "in_blink": False, "completed_blink_duration_ms": 100 if timestamp == 65 else None,
        }
        eye = Mock()
        eye.update.side_effect = lambda image, timestamp: {
            "face_detected": True, "calibrated": True,
            "sample_valid": timestamp == 65, "reduction_ratio": 2,
        }
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("camera.video_file.VideoFile",
                                      return_value=contextlib.nullcontext(source)))
            stack.enter_context(patch("detectors.blink_detector.BlinkDetector", return_value=blink))
            stack.enter_context(patch("detectors.blink_duration.BlinkDuration", return_value=duration))
            stack.enter_context(patch("detectors.eye_opening_reduction.EyeOpeningReduction", return_value=eye))
            if plot_error:
                stack.enter_context(patch("utils.reporting.save_graphs", side_effect=RuntimeError("plot failed")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            status = analyze_main([str(base.with_suffix(".mp4")), "--csv", str(base),
                                   "--user-id", "u1", "--no-display"])
        blink.close.assert_called_once()
        eye.close.assert_called_once()
        return status

    def test_video_entry_point_automatically_writes_both_csvs_and_graphs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "user1.csv"
            self.assertEqual(self.run_analysis(base), 0)
            paths = report_paths(base)
            self.assertTrue(all(path.exists() for path in paths.all_paths))
            raw = read_measurements(paths.raw)
            self.assertEqual(len(raw), 2)
            self.assertTrue(raw[-1].is_partial)
            with paths.summary.open(encoding="utf-8-sig", newline="") as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual(len(summary), 2)
            self.assertEqual(summary[-1]["측정 상태"], "부분 구간")

    def test_plot_failure_keeps_both_completed_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "user1.csv"
            self.assertEqual(self.run_analysis(base, plot_error=True), 1)
            paths = report_paths(base)
            self.assertEqual(len(read_measurements(paths.raw)), 2)
            with paths.summary.open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)


if __name__ == "__main__":
    unittest.main()
