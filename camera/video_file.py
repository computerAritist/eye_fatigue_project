"""Sequential OpenCV file input. This module never imports Raspberry Pi drivers."""

from dataclasses import dataclass
from pathlib import Path
import math

from utils.video_time import VideoClock


@dataclass
class VideoFrame:
    image: object
    timestamp: float
    index: int


class VideoFile:
    def __init__(self, path, timestamp_mode="auto", *, capture_factory=None):
        import cv2

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {self.path}")
        self.cap = (capture_factory or cv2.VideoCapture)(str(self.path))
        self.eof = False
        self.closed = False
        try:
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.path}")
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            self.clock = VideoClock(self.fps, timestamp_mode)
            count = float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.expected_frames = int(count) if math.isfinite(count) and count > 0 else 0
            self._position_property = cv2.CAP_PROP_POS_MSEC
        except BaseException:
            self.release()
            raise

    def read(self):
        if self.closed:
            raise RuntimeError("Video source is closed.")
        if self.eof:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.eof = True
            decoded = self.clock.frame_count
            if not decoded:
                raise RuntimeError("The video contains no decodable frames.")
            if self.expected_frames and decoded < self.expected_frames:
                raise RuntimeError(
                    f"Video decoding stopped early: {decoded}/{self.expected_frames} frames. "
                    "The file may be damaged, or its frame-count metadata inaccurate."
                )
            return None
        position_ms = float(self.cap.get(self._position_property))
        timestamp = self.clock.update(position_ms)
        return VideoFrame(frame, timestamp, self.clock.frame_count - 1)

    @property
    def end_time(self):
        return self.clock.end_time

    def release(self):
        if not self.closed:
            self.cap.release()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
