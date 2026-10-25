from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2

from analyze_video import analyze_stream
from camera.video_file import VideoFile, VideoFrame


class Image:
    def __init__(self, marks=None):
        self.marks = list(marks or [])

    def copy(self):
        return Image(self.marks)


class Source:
    def __init__(self, timestamps, end_time, error=None):
        self.samples = iter(VideoFrame(Image(), t, i) for i, t in enumerate(timestamps))
        self.end_time = end_time
        self.read_count = 0
        self.error = error

    def read(self):
        self.read_count += 1
        sample = next(self.samples, None)
        if sample is None and self.error is not None:
            raise self.error
        return sample


class Blink:
    def __init__(self):
        self.times = []
        self.reset_count = 0

    def reset_tracking(self):
        self.reset_count += 1

    def update(self, frame, timestamp):
        self.times.append(timestamp)
        frame.marks.append("drawn")
        return {"frame": frame, "ear": 0.3, "face_detected": timestamp != 65.1,
                "blink_event": timestamp in (5, 65)}


class Duration(Blink):
    def update(self, ear, timestamp, face_detected):
        self.times.append(timestamp)
        return {"in_blink": False, "completed_blink_duration_ms": (
            100 if timestamp in (5, 65) else None
        )}


class Eye(Blink):
    def __init__(self, calibration_time=5):
        super().__init__()
        self.calibration_time = calibration_time
        self.unmodified_inputs = True

    def update(self, frame, timestamp):
        self.times.append(timestamp)
        self.unmodified_inputs &= not frame.marks
        return {"calibrated": timestamp >= self.calibration_time,
                "face_detected": True,
                "sample_valid": timestamp > self.calibration_time,
                "reduction_ratio": 2}


class Writer:
    def __init__(self):
        self.rows = []

    def write(self, row):
        self.rows.append(row)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.blink, self.duration, self.eye = Blink(), Duration(), Eye()
        self.writer = Writer()

    def analyze(self, source, on_frame=None):
        return analyze_stream(source, self.blink, self.duration, self.eye,
                              self.writer, "u1", on_frame)

    def test_shared_clock_calibration_boundary_events_and_normal_eof(self):
        times = [0, 4.9, 5, 64.9, 65, 65.1]
        source = Source(times, 65.2)
        summary = self.analyze(source)
        self.assertEqual(self.blink.times, times)
        self.assertEqual(self.duration.times, times)
        self.assertEqual(self.eye.times, times)
        self.assertTrue(self.eye.unmodified_inputs)
        self.assertEqual(self.blink.reset_count, 1)
        self.assertEqual(self.duration.reset_count, 1)
        self.assertEqual(source.read_count, len(times) + 1)
        self.assertEqual(summary.end_reason, "eof")
        self.assertEqual(summary.window_count, 2)
        first, last = self.writer.rows
        self.assertEqual(first.blink_count, 0)  # calibration event discarded
        self.assertEqual(first.duration_sample_count, 0)
        self.assertEqual(last.blink_count, 1)
        self.assertEqual(last.duration_sample_count, 1)
        self.assertEqual(last.face_detection_success_pct, 50)
        self.assertTrue(last.is_partial)

    def test_live_mean_is_current_window_not_previous_minute(self):
        seen = []
        self.analyze(Source([5, 65], 65.1), lambda ctx: seen.append(ctx["measurement"]))
        self.assertIsNone(seen[0].mean_blink_duration_ms)
        self.assertEqual(seen[1].mean_blink_duration_ms, 100)

    def test_eor_not_collected_while_duration_detector_is_in_blink(self):
        original = self.duration.update

        def update(*args, **kwargs):
            result = original(*args, **kwargs)
            result["in_blink"] = True
            return result

        with patch.object(self.duration, "update", side_effect=update):
            self.analyze(Source([5, 6], 6.1))
        self.assertEqual(self.writer.rows[0].eor_sample_count, 0)

    def test_no_calibration_means_no_measurement_rows(self):
        summary = self.analyze(Source([0, 1, 2], 3))
        self.assertFalse(summary.calibrated)
        self.assertEqual(summary.window_count, 0)
        self.assertEqual(self.writer.rows, [])

    def test_user_quit_saves_partial(self):
        summary = self.analyze(Source([5, 6], 5.1), lambda ctx: False)
        self.assertEqual(summary.end_reason, "user_quit")
        self.assertEqual(self.writer.rows[0].end_reason, "user_quit")
        self.assertTrue(self.writer.rows[0].is_partial)

    def test_interrupt_saves_partial(self):
        summary = self.analyze(Source([5, 6], 6.1, KeyboardInterrupt()))
        self.assertEqual(summary.end_reason, "interrupted")
        self.assertEqual(self.writer.rows[0].end_reason, "interrupted")

    def test_decode_error_preserves_completed_row_and_invalid_partial(self):
        with self.assertRaisesRegex(RuntimeError, "decode"):
            self.analyze(Source([5, 65], 65.1, RuntimeError("decode")))
        self.assertEqual(len(self.writer.rows), 2)
        self.assertFalse(self.writer.rows[0].is_partial)
        self.assertEqual(self.writer.rows[1].end_reason, "error")
        self.assertFalse(self.writer.rows[1].measurement_valid)


class FakeCapture:
    def __init__(self, *, fps=10, count=2, positions=(0, 100), opened=True):
        self.fps, self.count = fps, count
        self.positions = positions
        self.index = 0
        self.opened = opened
        self.release_count = 0

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.count
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.positions[self.index - 1]
        return 0

    def read(self):
        if self.index >= len(self.positions):
            return False, None
        self.index += 1
        return True, Image()

    def release(self):
        self.release_count += 1


class VideoFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "placeholder.mp4"
        self.path.touch()  # Capture is a fake; this is not an encoded video.

    def source(self, capture):
        return VideoFile(self.path, "timestamps", capture_factory=lambda path: capture)

    def test_eof_returns_none_and_release_is_idempotent(self):
        cap = FakeCapture()
        with self.source(cap) as source:
            self.assertEqual(source.read().timestamp, 0)
            self.assertAlmostEqual(source.read().timestamp, 0.1)
            self.assertIsNone(source.read())
            self.assertIsNone(source.read())
            self.assertAlmostEqual(source.end_time, 0.2)
        source.release()
        self.assertEqual(cap.release_count, 1)

    def test_early_decode_failure_is_not_reported_as_normal_eof(self):
        with self.source(FakeCapture(count=3)) as source:
            source.read()
            source.read()
            with self.assertRaisesRegex(RuntimeError, "stopped early"):
                source.read()

    def test_empty_stream(self):
        with self.source(FakeCapture(count=0, positions=())) as source:
            with self.assertRaisesRegex(RuntimeError, "no decodable"):
                source.read()

    def test_bad_fps_or_open_failure_releases_capture(self):
        for cap in (FakeCapture(fps=0), FakeCapture(opened=False)):
            with self.subTest(cap=cap), self.assertRaises((ValueError, RuntimeError)):
                self.source(cap)
            self.assertEqual(cap.release_count, 1)


if __name__ == "__main__":
    unittest.main()
