from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from detectors.blink_detector import BlinkDetector
from detectors.blink_duration import BlinkDuration
from detectors.eye_opening_reduction import EyeOpeningReduction


def fake_face():
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    shape = [(0, 0), (3, -2), (7, -2), (10, 0), (7, 2), (3, 2)]
    for indices, origin in ((BlinkDetector.LEFT_EYE, 40), (BlinkDetector.RIGHT_EYE, 100)):
        for index, (x, y) in zip(indices, shape):
            landmarks[index] = SimpleNamespace(x=(origin + x * 4) / 200,
                                               y=(80 + y * 4) / 200)
    return SimpleNamespace(landmark=landmarks)


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("builtins.print"))
        self.mesh = Mock()
        self.mesh.process.return_value = SimpleNamespace(multi_face_landmarks=[fake_face()])
        self.enterContext(patch("mediapipe.solutions.face_mesh.FaceMesh", return_value=self.mesh))
        self.frame = np.zeros((200, 200, 3), dtype=np.uint8)

    def blink_step(self, detector, ear, timestamp):
        with patch.object(detector, "eye_aspect_ratio", return_value=ear):
            return detector.update(self.frame.copy(), timestamp=timestamp)

    def test_blink_thresholds_duration_limits_and_event_once(self):
        for duration, minimum_ear, expected in (
            (0.059, 0.16, 0), (0.06, 0.16, 1), (1.0, 0.16, 1),
            (1.001, 0.16, 0), (0.2, 0.19, 0), (0.2, 0.18, 1), (0.2, 0.181, 0),
        ):
            with self.subTest(duration=duration, minimum_ear=minimum_ear):
                detector = BlinkDetector(aggregate_minutes=False)
                self.blink_step(detector, minimum_ear, 0)
                result = self.blink_step(detector, 0.23, duration)
                self.assertEqual(result["total_blinks"], expected)
                self.assertEqual(result["blink_event"], bool(expected))
                result = self.blink_step(detector, 0.3, duration + 0.1)
                self.assertFalse(result["blink_event"])
                self.assertEqual(result["total_blinks"], expected)

    def test_blink_face_loss_cancels_unfinished_event(self):
        detector = BlinkDetector(aggregate_minutes=False)
        self.blink_step(detector, 0.16, 0)
        self.mesh.process.return_value = SimpleNamespace(multi_face_landmarks=None)
        lost = detector.update(self.frame, timestamp=0.1)
        self.assertFalse(lost["eye_closed"])
        self.mesh.process.return_value = SimpleNamespace(multi_face_landmarks=[fake_face()])
        opened = self.blink_step(detector, 0.3, 0.2)
        self.assertEqual(opened["total_blinks"], 0)

    def test_duration_thresholds_limits_and_event_once(self):
        for seconds, expected in ((0.079, False), (0.08, True), (2, True), (2.001, False)):
            with self.subTest(seconds=seconds):
                detector = BlinkDuration(aggregate_minutes=False)
                detector.update(0.20, timestamp=0)
                result = detector.update(0.24, timestamp=seconds)
                if expected:
                    self.assertAlmostEqual(result["completed_blink_duration_ms"], seconds * 1000)
                else:
                    self.assertIsNone(result["completed_blink_duration_ms"])
                self.assertEqual(detector.blink_durations, [])  # bounded memory in video mode
                self.assertIsNone(detector.update(0.3, timestamp=seconds + 0.1)
                                  ["completed_blink_duration_ms"])

    def test_duration_face_loss_does_not_bridge_missing_frames(self):
        for missing_ear, face in ((0, None), (0.18, False)):
            with self.subTest(missing_ear=missing_ear):
                detector = BlinkDuration(aggregate_minutes=False)
                detector.update(0.16, timestamp=0)
                lost = detector.update(missing_ear, timestamp=0.1, face_detected=face)
                self.assertFalse(lost["in_blink"])
                reopened = detector.update(0.3, timestamp=0.3)
                self.assertIsNone(reopened["completed_blink_duration_ms"])

    def test_original_duration_call_uses_wall_time_and_minute_mean(self):
        with patch("detectors.blink_duration.time.time", side_effect=[100, 101, 101.2, 161]):
            detector = BlinkDuration()
            detector.update(0.16)
            detector.update(0.3)
            result = detector.update(0.3)
        self.assertTrue(result["measurement_complete"])
        self.assertEqual(result["blink_count"], 1)
        self.assertAlmostEqual(result["mean_blink_duration_ms"], 200)

    def test_original_blink_call_uses_wall_time(self):
        with patch("detectors.blink_detector.time.time", side_effect=[100, 101, 101.2]):
            detector = BlinkDetector()
            with patch.object(detector, "eye_aspect_ratio", return_value=0.16):
                detector.update(self.frame.copy())
            with patch.object(detector, "eye_aspect_ratio", return_value=0.3):
                result = detector.update(self.frame.copy())
        self.assertEqual(result["total_blinks"], 1)

    def test_close_thresholds_are_strict(self):
        blink = BlinkDetector(aggregate_minutes=False)
        self.assertFalse(self.blink_step(blink, 0.20, 0)["eye_closed"])
        duration = BlinkDuration(aggregate_minutes=False)
        self.assertFalse(duration.update(0.21, timestamp=0)["in_blink"])

    def test_original_eye_update_call_uses_wall_time(self):
        eye = EyeOpeningReduction()
        eye.eye_aspect_ratio = Mock(return_value=0.3)
        eye.calculate_eye_size = Mock(return_value=0.28)
        with patch("detectors.eye_opening_reduction.time.time", side_effect=[100, 104.99, 105]):
            self.assertFalse(eye.update(self.frame)["calibrated"])
            self.assertFalse(eye.update(self.frame)["calibrated"])
            self.assertTrue(eye.update(self.frame)["calibrated"])

    def test_normalized_contour_area_and_ear_formulas_unchanged(self):
        detector = EyeOpeningReduction()
        eye = [(0, 0), (3, -2), (7, -2), (10, 0), (7, 2), (3, 2)]
        self.assertAlmostEqual(detector.calculate_eye_size(eye), 0.28)
        self.assertAlmostEqual(detector.eye_aspect_ratio(eye), 0.4)
        scaled = [(x * 10, y * 10) for x, y in eye]
        self.assertAlmostEqual(detector.calculate_eye_size(scaled), 0.28)

    def test_five_video_seconds_calibration_and_fifteen_valid_sample_smoothing(self):
        detector = EyeOpeningReduction()
        detector.eye_aspect_ratio = Mock(return_value=0.3)
        detector.calculate_eye_size = Mock(return_value=0.28)
        self.assertFalse(detector.update(self.frame, timestamp=0)["calibrated"])
        self.assertFalse(detector.update(self.frame, timestamp=4.99)["calibrated"])
        calibrated = detector.update(self.frame, timestamp=5)
        self.assertTrue(calibrated["calibrated"])
        self.assertFalse(calibrated["sample_valid"])
        self.assertAlmostEqual(calibrated["baseline_eye_size"], 0.28)
        detector.calculate_eye_size.return_value = 0.21
        sample = detector.update(self.frame, timestamp=5.1)
        self.assertTrue(sample["sample_valid"])
        self.assertAlmostEqual(sample["reduction_ratio"], 100 * 0.25 / 15)
        detector.eye_aspect_ratio.return_value = 0.23
        held = detector.update(self.frame, timestamp=5.2)
        self.assertFalse(held["sample_valid"])
        self.assertEqual(held["reduction_ratio"], sample["reduction_ratio"])
        self.mesh.process.return_value = SimpleNamespace(multi_face_landmarks=None)
        missing = detector.update(self.frame, timestamp=5.3)
        self.assertFalse(missing["face_detected"])
        self.assertFalse(missing["sample_valid"])


if __name__ == "__main__":
    unittest.main()
