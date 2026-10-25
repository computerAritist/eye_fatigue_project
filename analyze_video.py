"""Analyze a recorded video without importing GPIO, PWM, LCD, or I2C modules."""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import sys
import time

from utils.measurement_window import MeasurementWindows


@dataclass(frozen=True)
class AnalysisSummary:
    frame_count: int
    window_count: int
    calibrated: bool
    end_reason: str
    end_sec: float


def analyze_stream(source, blink_detector, blink_duration, eye_detector, writer,
                   user_id, on_frame=None):
    """Injectable analysis loop; callbacks return False to stop after this frame.

    Face success requires both independent FaceMesh passes. Calibration frames
    are excluded from measurement. Completed events belong to [start, end)
    according to their completion timestamp, including cross-boundary blinks.
    """
    windows = None
    frame_count = 0
    reason = "eof"
    try:
        while True:
            sample = source.read()
            if sample is None:
                break
            frame_count += 1
            # BlinkDetector draws on its input; give EOR the untouched image.
            eye_frame = sample.image.copy()
            blink = blink_detector.update(sample.image, timestamp=sample.timestamp)
            duration = blink_duration.update(
                blink["ear"], timestamp=sample.timestamp,
                face_detected=blink["face_detected"],
            )
            eye = eye_detector.update(eye_frame, timestamp=sample.timestamp)

            just_calibrated = windows is None and eye["calibrated"]
            if just_calibrated:
                windows = MeasurementWindows(user_id, sample.timestamp, writer.write)
                # Do not carry a closure started during calibration into results.
                blink_detector.reset_tracking()
                blink_duration.reset_tracking()
            if windows is not None:
                windows.observe(
                    sample.timestamp,
                    face_detected=blink["face_detected"] and eye["face_detected"],
                    blink_event=blink["blink_event"] and not just_calibrated,
                    blink_duration_ms=(duration["completed_blink_duration_ms"]
                                       if not just_calibrated else None),
                    eor=(eye["reduction_ratio"] if eye["sample_valid"]
                         and not duration["in_blink"] else None),
                    calibrated=eye["calibrated"],
                )
            if on_frame is not None:
                keep_going = on_frame({
                    "sample": sample, "blink": blink, "duration": duration,
                    "eye": eye, "measurement": (
                        windows.snapshot(sample.timestamp) if windows is not None else None
                    ),
                })
                if keep_going is False:
                    reason = "user_quit"
                    break
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception:
        reason = "error"
        raise
    finally:
        if windows is not None:
            windows.finish(source.end_time, end_reason=reason)
    return AnalysisSummary(
        frame_count, windows.completed_count if windows is not None else 0,
        windows is not None, reason, source.end_time,
    )


def format_time(seconds):
    total_ms = round(seconds * 1000)
    seconds, milliseconds = divmod(total_ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class VideoDisplay:
    def __init__(self, *, realtime=False, display_width=1280):
        import cv2

        self.cv2 = cv2
        self.realtime = realtime
        self.display_width = display_width
        self.wall_start = None
        self.video_start = None
        self.name = "Recorded Video - Eye Fatigue (Q: quit)"
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)

    def render(self, context):
        """Compose a separate responsive panel above the unoccluded video.

        BlinkDetector already draws the original six points per eye. Rendering
        neither adds full-face landmarks nor writes into the detector's frame.
        """
        import numpy as np

        cv2 = self.cv2
        sample = context["sample"]
        blink, eye = context["blink"], context["eye"]
        measurement = context["measurement"]
        frame = blink["frame"]
        # Only presentation is resized. Keep a readable minimum canvas width for
        # small or portrait videos, padding rather than stretching the image.
        height, width = frame.shape[:2]
        factor = min(1.0, self.display_width / width)
        video = cv2.resize(frame, (max(1, round(width * factor)), max(1, round(height * factor))))
        canvas_width = max(600, video.shape[1])
        padding = canvas_width - video.shape[1]
        video = cv2.copyMakeBorder(video, 0, 0, padding // 2, padding - padding // 2,
                                   cv2.BORDER_CONSTANT, value=(18, 18, 18))
        unit = min(1.4, max(0.8, canvas_width / 1000))
        panel = np.full((round(228 * unit), canvas_width, 3), (28, 24, 22), dtype=np.uint8)
        margin, gap = round(18 * unit), round(12 * unit)
        font = cv2.FONT_HERSHEY_SIMPLEX
        foreground, muted, accent = (245, 245, 245), (180, 174, 168), (230, 202, 90)

        def text(label, x, y, size, color=foreground, max_width=None):
            scale = size * unit
            thickness = max(1, round(scale * 1.4))
            if max_width is not None:
                text_width = cv2.getTextSize(label, font, scale, thickness)[0][0]
                if text_width > max_width:
                    scale *= max_width / text_width
            cv2.putText(panel, label, (x, round(y * unit)), font, scale,
                        color, thickness, cv2.LINE_AA)

        if measurement is None:
            heading = "CALIBRATING"
            interval = "Measurement starts after eye-size calibration"
            count = bd = eor = "N/A"
        else:
            heading = f"WINDOW {measurement.window_index:02d}"
            interval = (f"{format_time(measurement.start_sec)} - "
                        f"{format_time(measurement.start_sec + 60)}")
            count = str(measurement.blink_count)
            bd = (f"{measurement.mean_blink_duration_ms:.1f} ms"
                  if measurement.mean_blink_duration_ms is not None else "N/A")
            eor = (f"{measurement.mean_eye_opening_reduction_pct:.2f}%"
                   if measurement.mean_eye_opening_reduction_pct is not None else "N/A")
        text(heading, margin, 37, 0.85, accent, canvas_width - 2 * margin)
        text(interval, margin, 65, 0.46, muted, canvas_width - 2 * margin)
        card_width = (canvas_width - 2 * margin - 2 * gap) // 3
        for index, (label, value) in enumerate((
            ("BLINK COUNT", count), ("MEAN CLOSURE", bd), ("MEAN EYE REDUCTION", eor),
        )):
            left = margin + index * (card_width + gap)
            cv2.rectangle(panel, (left, round(82 * unit)),
                          (left + card_width, round(179 * unit)), (43, 37, 32), -1)
            inset = round(12 * unit)
            text(label, left + inset, 108, 0.43, muted, card_width - 2 * inset)
            text(value, left + inset, 157, 1.0, foreground, card_width - 2 * inset)

        blink_face = blink.get("face_detected", False)
        eye_face = eye.get("face_detected", False)
        face_detected = blink_face and eye_face
        ear = f"{blink['ear']:.3f}" if blink_face else "N/A"
        face_label = "detected" if face_detected else "not detected"
        face_color = (135, 205, 145) if face_detected else (135, 155, 240)
        text(f"EAR: {ear}   Face: {face_label}", margin, 210, 0.43, face_color,
             canvas_width // 2 - margin)
        video_time = f"Video {format_time(sample.timestamp)}"
        time_width = cv2.getTextSize(video_time, font, 0.43 * unit, 1)[0][0]
        text(video_time, canvas_width - margin - time_width, 210, 0.43, muted,
             canvas_width // 2 - margin)
        return np.vstack((panel, video))

    def __call__(self, context):
        cv2 = self.cv2
        sample = context["sample"]
        display = self.render(context)
        cv2.imshow(self.name, display)
        if self.wall_start is None:
            self.wall_start = time.monotonic()
            self.video_start = sample.timestamp
        target = self.wall_start + sample.timestamp - self.video_start
        while True:
            remaining = target - time.monotonic() if self.realtime else 0
            delay = max(1, min(50, round(remaining * 1000)))
            if cv2.waitKey(delay) & 0xFF in (ord("q"), ord("Q"), 27):
                return False
            if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                return False
            if not self.realtime or time.monotonic() >= target:
                return True

    def close(self):
        self.cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input MP4/video path")
    parser.add_argument("--user-id", required=True, help="User ID saved in every CSV row")
    parser.add_argument("--csv", required=True, type=Path,
                        help="Output base name: user1.csv creates user1_raw.csv, user1_summary.csv and PNGs")
    parser.add_argument("--timestamp-mode", choices=("auto", "timestamps", "fps"),
                        default="auto", help="auto falls back to constant-FPS time with a warning")
    parser.add_argument("--no-display", action="store_true", help="Analyze without GUI")
    parser.add_argument("--realtime", action="store_true", help="Pace the display to video time")
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output CSV")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.user_id.strip():
        parser.error("--user-id must not be empty")
    if args.display_width <= 0:
        parser.error("--display-width must be greater than zero")
    if (args.video.resolve() == args.csv.resolve()
            or (args.video.exists() and args.csv.exists() and args.video.samefile(args.csv))):
        parser.error("The input video and output CSV must be different files")
    try:
        from utils.reporting import AnalysisReports, check_outputs, report_paths

        check_outputs(report_paths(args.csv).all_paths, overwrite=args.overwrite, source=args.video)
        # Lazy imports keep --help and pure aggregation tests independent of MediaPipe.
        from camera.video_file import VideoFile
        from detectors.blink_detector import BlinkDetector
        from detectors.blink_duration import BlinkDuration
        from detectors.eye_opening_reduction import EyeOpeningReduction

        with ExitStack() as stack:
            source = stack.enter_context(VideoFile(args.video, args.timestamp_mode))
            print(f"[VIDEO] FPS={source.fps:.6f}; metadata frames={source.expected_frames}; "
                  f"timestamp mode={args.timestamp_mode}")
            writer = stack.enter_context(AnalysisReports(args.csv, overwrite=args.overwrite))
            blink = BlinkDetector(aggregate_minutes=False)
            stack.callback(blink.close)
            duration = BlinkDuration(aggregate_minutes=False)
            eye = EyeOpeningReduction()
            stack.callback(eye.close)
            display = None
            if not args.no_display:
                display = VideoDisplay(realtime=args.realtime, display_width=args.display_width)
                stack.callback(display.close)
            try:
                summary = analyze_stream(source, blink, duration, eye, writer, args.user_id, display)
            except BaseException:
                try:
                    writer.finalize()
                except Exception as report_error:
                    print(f"[REPORT ERROR] {report_error}; completed raw rows are preserved.",
                          file=sys.stderr)
                raise
            else:
                # Both CSVs are already flushed, even if plotting fails after analysis.
                writer.finalize()
            print(f"[DONE] frames={summary.frame_count}; windows={summary.window_count}; "
                  f"end={format_time(summary.end_sec)}; reason={summary.end_reason}; "
                  f"FPS fallback={source.clock.used_fps_fallback}; raw CSV={writer.paths.raw}")
            if summary.end_reason == "interrupted":
                return 130
            if not summary.calibrated:
                print("[WARNING] Calibration did not complete; CSV contains only the header.",
                      file=sys.stderr)
                return 2
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
