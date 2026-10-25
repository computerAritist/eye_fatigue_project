"""One owner for half-open measurement windows and completed blink events."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class MeasurementResult:
    user_id: str
    window_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    blink_count: int
    mean_blink_duration_ms: float | None
    duration_sample_count: int
    mean_eye_opening_reduction_pct: float | None
    eor_sample_count: int
    face_detection_success_pct: float
    frame_count: int
    face_detected_count: int
    calibrated: bool
    is_partial: bool
    measurement_valid: bool
    end_reason: str

    def as_dict(self):
        return asdict(self)


class MeasurementWindows:
    """Assign a blink to its completion time, never resetting detector state."""

    def __init__(self, user_id, start_sec, sink, interval=60.0):
        if not math.isfinite(start_sec) or start_sec < 0:
            raise ValueError("Invalid measurement start time.")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Invalid measurement interval.")
        self.user_id = user_id
        self.origin = float(start_sec)
        self.interval = float(interval)
        self.sink = sink
        self.window_index = 1
        self.completed_count = 0
        self.last_timestamp = None
        self.finished = False
        self._reset_samples()

    def _reset_samples(self):
        self.frame_count = 0
        self.face_detected_count = 0
        self.blink_count = 0
        self.duration_sum = 0.0
        self.duration_sample_count = 0
        self.eor_sum = 0.0
        self.eor_sample_count = 0
        self.calibrated = True

    @property
    def start_sec(self):
        return self.origin + (self.window_index - 1) * self.interval

    @property
    def end_sec(self):
        return self.origin + self.window_index * self.interval

    def _advance(self, timestamp):
        while timestamp >= self.end_sec:
            self._emit(self.end_sec, False, "interval")
            self.window_index += 1
            self._reset_samples()

    def observe(self, timestamp, *, face_detected, blink_event=False,
                blink_duration_ms=None, eor=None, calibrated=True):
        if self.finished:
            raise RuntimeError("Measurement already finalized.")
        if (not math.isfinite(timestamp) or timestamp < self.start_sec
                or (self.last_timestamp is not None and timestamp <= self.last_timestamp)):
            raise ValueError("Frame times must be strictly increasing within the measurement.")
        self._advance(timestamp)
        self.last_timestamp = timestamp
        self.frame_count += 1
        self.face_detected_count += bool(face_detected)
        self.calibrated = self.calibrated and calibrated
        self.blink_count += bool(blink_event)
        if blink_duration_ms is not None:
            if not math.isfinite(blink_duration_ms) or blink_duration_ms < 0:
                raise ValueError("Invalid blink duration.")
            self.duration_sum += blink_duration_ms
            self.duration_sample_count += 1
        if eor is not None:
            if not math.isfinite(eor) or not 0 <= eor <= 100:
                raise ValueError("Invalid eye opening reduction.")
            self.eor_sum += eor
            self.eor_sample_count += 1

    def snapshot(self, end_sec=None, *, is_partial=False, end_reason="interval"):
        end_sec = self.end_sec if end_sec is None else end_sec
        mean_bd = (self.duration_sum / self.duration_sample_count
                   if self.duration_sample_count else None)
        mean_eor = self.eor_sum / self.eor_sample_count if self.eor_sample_count else None
        valid = bool(self.calibrated and self.face_detected_count
                     and self.duration_sample_count and self.eor_sample_count
                     and end_reason != "error")
        return MeasurementResult(
            user_id=self.user_id, window_index=self.window_index,
            start_sec=self.start_sec, end_sec=end_sec, duration_sec=end_sec - self.start_sec,
            blink_count=self.blink_count, mean_blink_duration_ms=mean_bd,
            duration_sample_count=self.duration_sample_count,
            mean_eye_opening_reduction_pct=mean_eor, eor_sample_count=self.eor_sample_count,
            face_detection_success_pct=(100.0 * self.face_detected_count / self.frame_count
                                        if self.frame_count else 0.0),
            frame_count=self.frame_count, face_detected_count=self.face_detected_count,
            calibrated=self.calibrated, is_partial=is_partial,
            measurement_valid=valid, end_reason=end_reason,
        )

    def _emit(self, end_sec, is_partial, end_reason):
        self.sink(self.snapshot(end_sec, is_partial=is_partial, end_reason=end_reason))
        self.completed_count += 1

    def finish(self, end_sec, end_reason="eof"):
        if self.finished:
            return
        if (not math.isfinite(end_sec) or end_sec < self.origin
                or (self.last_timestamp is not None and end_sec < self.last_timestamp)):
            raise ValueError("Invalid end time.")
        # Accommodate floating-point arithmetic at an exact final frame boundary.
        if math.isclose(end_sec, self.end_sec, abs_tol=1e-8, rel_tol=0):
            end_sec = self.end_sec
        self._advance(end_sec)
        if end_sec > self.start_sec and self.frame_count:
            self._emit(end_sec, True, end_reason)
        self.finished = True
