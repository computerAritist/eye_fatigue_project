"""Video time independent of processing speed; no camera or hardware imports."""

import math
import warnings


class VideoClock:
    def __init__(self, fps, mode="auto"):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS must be finite and greater than zero.")
        if mode not in {"auto", "timestamps", "fps"}:
            raise ValueError("Unknown timestamp mode.")
        self.fps = float(fps)
        self.mode = mode
        self.frame_count = 0
        self.last_timestamp = None
        self.last_interval = 1.0 / self.fps
        self._origin = None
        self._fps_anchor = None
        self._anchor_index = 0
        self.used_fps_fallback = mode == "fps"

    def update(self, position_ms):
        index = self.frame_count
        if self.used_fps_fallback:
            timestamp = self._fps_time(index)
        else:
            valid = math.isfinite(position_ms) and position_ms >= 0
            if valid:
                raw = position_ms / 1000.0
                if self._origin is None:
                    self._origin = raw
                timestamp = raw - self._origin
                valid = self.last_timestamp is None or timestamp > self.last_timestamp
            if not valid:
                if self.mode == "timestamps":
                    raise ValueError(
                        f"Missing or non-increasing video timestamp at frame {index}. "
                        "Use --timestamp-mode fps only for a known constant-FPS video."
                    )
                warnings.warn(
                    f"Invalid video timestamp at frame {index}; switching to FPS time. "
                    "This fallback assumes constant FPS. For variable-FPS input, "
                    "use --timestamp-mode timestamps and repair the timestamps.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.used_fps_fallback = True
                self._anchor_index = index
                self._fps_anchor = (
                    0.0 if self.last_timestamp is None
                    else self.last_timestamp + 1.0 / self.fps
                )
                timestamp = self._fps_time(index)
        if self.last_timestamp is not None:
            self.last_interval = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        self.frame_count += 1
        return timestamp

    def _fps_time(self, index):
        if self._fps_anchor is None:
            return index / self.fps
        return self._fps_anchor + (index - self._anchor_index) / self.fps

    @property
    def end_time(self):
        """End of the last decoded frame; final VFR frame length is an estimate."""
        if self.last_timestamp is None:
            return 0.0
        return self.last_timestamp + self.last_interval
