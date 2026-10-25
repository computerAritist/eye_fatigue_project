import csv
import math
from pathlib import Path
import tempfile
import unittest
import warnings

from utils.logger import MeasurementCSV
from utils.measurement_window import MeasurementWindows
from utils.video_time import VideoClock


class VideoClockTests(unittest.TestCase):
    def test_fps_time_and_frame_end(self):
        clock = VideoClock(30, "fps")
        times = [clock.update(float("nan")) for _ in range(1800)]
        self.assertEqual(times[0], 0)
        self.assertAlmostEqual(times[-1], 60 - 1 / 30)
        self.assertAlmostEqual(clock.end_time, 60)

    def test_variable_frame_timestamps_and_nonzero_origin(self):
        clock = VideoClock(30, "timestamps")
        for position, expected in zip((1000, 1030, 1080, 1100), (0, 0.03, 0.08, 0.10)):
            self.assertAlmostEqual(clock.update(position), expected)
        self.assertAlmostEqual(clock.end_time, 0.12)

    def test_auto_fallback_warns_once_and_stays_monotonic(self):
        clock = VideoClock(25)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            times = [clock.update(ms) for ms in (0, 40, 40, 0, math.nan)]
        self.assertEqual(len(caught), 1)
        self.assertEqual(times, [0, 0.04, 0.08, 0.12, 0.16])
        self.assertTrue(clock.used_fps_fallback)

    def test_strict_mode_rejects_missing_or_nonincreasing_timestamps(self):
        for bad in (0, -1, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                clock = VideoClock(30, "timestamps")
                clock.update(0)
                with self.assertRaises(ValueError):
                    clock.update(bad)

    def test_invalid_fps(self):
        for fps in (0, -30, math.nan, math.inf):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                VideoClock(fps)


class MeasurementWindowTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        self.windows = MeasurementWindows("participant-01", 5, self.rows.append)

    def test_half_open_boundary_and_cross_boundary_event(self):
        # Closure begins before 65, completes exactly at 65: second window only.
        self.windows.observe(5, face_detected=True, eor=1)
        self.windows.observe(64.9, face_detected=True, eor=3)
        self.windows.observe(65, face_detected=True, blink_event=True,
                             blink_duration_ms=100, eor=5)
        self.windows.observe(65.1, face_detected=True, eor=7)
        self.windows.finish(65.2)
        self.windows.finish(65.2)
        self.assertEqual(len(self.rows), 2)
        first, second = self.rows
        self.assertEqual((first.start_sec, first.end_sec, first.blink_count), (5, 65, 0))
        self.assertFalse(first.is_partial)
        self.assertEqual(first.mean_eye_opening_reduction_pct, 2)
        self.assertEqual(second.blink_count, 1)
        self.assertEqual(second.duration_sample_count, 1)
        self.assertEqual(second.mean_blink_duration_ms, 100)
        self.assertEqual(second.mean_eye_opening_reduction_pct, 6)
        self.assertTrue(second.is_partial)

    def test_hour_after_five_second_calibration(self):
        for timestamp in range(5, 3600):
            self.windows.observe(timestamp, face_detected=True, eor=2)
        self.windows.finish(3600)
        self.assertEqual(len(self.rows), 60)
        self.assertEqual(sum(not row.is_partial for row in self.rows), 59)
        self.assertEqual(self.rows[-1].start_sec, 3545)
        self.assertEqual(self.rows[-1].duration_sec, 55)
        self.assertTrue(self.rows[-1].is_partial)

    def test_exact_end_does_not_create_empty_trailing_row(self):
        self.windows.observe(5, face_detected=True)
        self.windows.observe(64.99, face_detected=True)
        self.windows.finish(65 - 1e-10)
        self.assertEqual(len(self.rows), 1)
        self.assertFalse(self.rows[0].is_partial)

    def test_means_sample_counts_face_ratio_and_validity(self):
        self.windows.observe(5, face_detected=True, blink_duration_ms=100, eor=2)
        self.windows.observe(6, face_detected=False)
        self.windows.observe(7, face_detected=True, blink_event=True,
                             blink_duration_ms=300, eor=4)
        self.windows.finish(8)
        row = self.rows[0]
        self.assertEqual(row.blink_count, 1)  # duration and count acceptance differ
        self.assertEqual(row.duration_sample_count, 2)
        self.assertEqual(row.mean_blink_duration_ms, 200)
        self.assertEqual(row.eor_sample_count, 2)
        self.assertEqual(row.mean_eye_opening_reduction_pct, 3)
        self.assertAlmostEqual(row.face_detection_success_pct, 200 / 3)
        self.assertTrue(row.measurement_valid)

    def test_missing_samples_stay_missing_instead_of_zero(self):
        self.windows.observe(5, face_detected=False)
        self.windows.finish(6)
        row = self.rows[0]
        self.assertIsNone(row.mean_blink_duration_ms)
        self.assertIsNone(row.mean_eye_opening_reduction_pct)
        self.assertFalse(row.measurement_valid)
        self.assertEqual(row.face_detection_success_pct, 0)

    def test_time_gaps_emit_invalid_empty_windows(self):
        self.windows.observe(5, face_detected=True)
        self.windows.observe(186, face_detected=True)
        self.windows.finish(187)
        self.assertEqual(len(self.rows), 4)
        self.assertEqual(self.rows[1].frame_count, 0)
        self.assertFalse(self.rows[1].measurement_valid)

    def test_duplicate_frame_times_are_rejected(self):
        self.windows.observe(5, face_detected=True)
        with self.assertRaises(ValueError):
            self.windows.observe(5, face_detected=True, blink_event=True)

    def test_abnormal_stop_marks_partial_measurement_invalid(self):
        self.windows.observe(5, face_detected=True, blink_duration_ms=100, eor=2)
        self.windows.finish(6, end_reason="error")
        self.assertFalse(self.rows[0].measurement_valid)
        self.assertEqual(self.rows[0].end_reason, "error")


class CSVTests(unittest.TestCase):
    def test_header_and_completed_rows_are_visible_before_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with MeasurementCSV(path) as writer:
                windows = MeasurementWindows("사용자-01", 5, writer.write)
                windows.observe(5, face_detected=True, blink_duration_ms=200, eor=2)
                windows.observe(65, face_detected=False)
                # The file is still open: visibility here depends on per-row flush.
                with path.open(encoding="utf-8-sig", newline="") as reader:
                    rows = list(csv.DictReader(reader))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["user_id"], "사용자-01")
                self.assertEqual(rows[0]["mean_blink_duration_ms"], "200.0")
                windows.finish(66)
            with path.open(encoding="utf-8-sig", newline="") as reader:
                rows = list(csv.DictReader(reader))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["mean_blink_duration_ms"], "")
            self.assertEqual(rows[1]["is_partial"], "True")
            with self.assertRaises(FileExistsError):
                MeasurementCSV(path)


if __name__ == "__main__":
    unittest.main()
