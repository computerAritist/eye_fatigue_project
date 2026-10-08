import cv2
import time
import traceback

from camera.camera import Camera
from controllers.led_controller import LEDController
from controllers.button_controller import ButtonController
from controllers.lcd_controller import LCDController

from detectors.blink_detector import BlinkDetector
from detectors.blink_duration import BlinkDuration
from detectors.eye_opening_reduction import EyeOpeningReduction
from detectors.fatigue_detector import FatigueDetector

from sensors.als_sensor import ALSSensor


MEASUREMENT_TIME = 60.0


def draw_text(frame, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(
        frame, text, (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2
    )
    return y + 28


def main():

    camera = blink_detector = blink_duration = None
    eye_detector = fatigue_detector = None
    led = als = buttons = lcd = None

    try:
        # ---------------- INIT ----------------

        camera = Camera()
        blink_detector = BlinkDetector()
        blink_duration = BlinkDuration(
            close_threshold=0.21,
            open_threshold=0.24
        )

        eye_detector = EyeOpeningReduction()
        fatigue_detector = FatigueDetector()

        led = LEDController(
            default_brightness=20,
            default_cct_mode=3
        )

        als = ALSSensor()
        buttons = ButtonController()
        lcd = LCDController()

        print("[INIT] ALL READY")

        # ---------------- SETTINGS ----------------

        mode = "AUTO"

        brightness_levels = [20, 40, 60, 80, 100]
        manual_brightness_index = 0

        current_brightness = 20
        current_cct_mode = 3
        cct_mode = 3

        auto_settings = {
            1: (100, 1),
            2: (80, 2),
            3: (60, 3),
            4: (40, 4),
            5: (20, 5)
        }

        status = "Measuring..."
        fatigue_level = None
        fatigue_score = 0.0

        br_level = bd_level = eor_level = 0

        measurement_started = False
        measurement_start = 0.0
        blink_start_count = 0

        eye_samples = []
        mean_eor = 0.0

        led.set_light(
            current_brightness,
            current_cct_mode
        )

        lcd.update(
            mode,
            status,
            current_brightness,
            current_cct_mode
        )

        print("[SYSTEM] Camera Window Start")

        # ---------------- MAIN LOOP ----------------

        while True:

            # ---------- BUTTON ----------

            event = buttons.get_event()

            if event == "MODE":

                mode = (
                    "MANUAL"
                    if mode == "AUTO"
                    else "AUTO"
                )

                if mode == "MANUAL":

                    manual_brightness_index = min(
                        range(len(brightness_levels)),
                        key=lambda i:
                        abs(
                            brightness_levels[i]
                            - current_brightness
                        )
                    )

                    cct_mode = current_cct_mode

                print(f"[MODE] {mode}")

            elif (
                event == "BRIGHTNESS"
                and mode == "MANUAL"
            ):

                manual_brightness_index = (
                    manual_brightness_index + 1
                ) % len(brightness_levels)

                current_brightness = (
                    brightness_levels[
                        manual_brightness_index
                    ]
                )

                led.set_light(
                    current_brightness,
                    current_cct_mode
                )

            elif (
                event == "CCT"
                and mode == "MANUAL"
            ):

                cct_mode = cct_mode % 6 + 1
                current_cct_mode = cct_mode

                led.set_light(
                    current_brightness,
                    current_cct_mode
                )

            # ---------- CAMERA ----------

            frame = camera.read()

            if frame is None:
                print("[CAMERA] Frame read failed")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            eye_frame = frame.copy()

            # ---------- BLINK ----------

            blink = blink_detector.update(frame)

            frame = blink.get("frame", frame)
            ear = blink.get("ear", 0.0)
            total_blinks = blink.get("total_blinks", 0)

            # ---------- BLINK DURATION ----------

            duration = blink_duration.update(ear)

            in_blink = duration.get(
                "in_blink",
                False
            )

            current_bd = duration.get(
                "current_blink_duration_ms",
                0.0
            )

            mean_bd = duration.get(
                "mean_blink_duration_ms",
                0.0
            )

            # ---------- EYE OPENING ----------

            eye = eye_detector.update(
                eye_frame
            )

            face_detected = eye.get(
                "face_detected",
                False
            )

            calibrated = eye.get(
                "calibrated",
                False
            )

            eor = eye.get(
                "reduction_ratio",
                0.0
            )

            # ---------- 60초 측정 시작 ----------

            if calibrated and not measurement_started:

                measurement_started = True
                measurement_start = time.time()

                blink_start_count = total_blinks
                eye_samples.clear()

                status = "Measuring..."

                print("[MEASURE] 60 sec START")

            remaining = MEASUREMENT_TIME

            # ---------- 60초 측정 ----------

            if measurement_started:

                elapsed = (
                    time.time()
                    - measurement_start
                )

                remaining = max(
                    0,
                    MEASUREMENT_TIME - elapsed
                )

                if (
                    face_detected
                    and not in_blink
                ):
                    eye_samples.append(eor)

                mean_eor = (
                    sum(eye_samples)
                    / len(eye_samples)
                    if eye_samples
                    else 0.0
                )

                # ---------- 측정 완료 ----------

                if elapsed >= MEASUREMENT_TIME:

                    blink_rate = (
                        total_blinks
                        - blink_start_count
                    )

                    result = fatigue_detector.update(
                        blink_rate,
                        mean_bd,
                        mean_eor
                    )

                    fatigue_level = result["level"]
                    fatigue_score = result["score"]

                    br_level = result["br_level"]
                    bd_level = result["bd_level"]
                    eor_level = result["eor_level"]

                    status = (
                        f"LEVEL {fatigue_level}"
                    )

                    print(
                        f"[RESULT] "
                        f"BR={blink_rate}/min | "
                        f"BD={mean_bd:.1f}ms | "
                        f"EOR={mean_eor:.1f}% | "
                        f"LEVEL={fatigue_level} | "
                        f"SCORE={fatigue_score:.2f}"
                    )

                    # AUTO LED
                    if mode == "AUTO":

                        (
                            current_brightness,
                            current_cct_mode
                        ) = auto_settings[
                            fatigue_level
                        ]

                        led.set_light(
                            current_brightness,
                            current_cct_mode
                        )

                    # 다음 60초
                    measurement_start = time.time()
                    blink_start_count = total_blinks
                    eye_samples.clear()

            # ---------- ALS ----------

            try:
                lux, cct = als.read()

            except Exception:
                lux, cct = 0, 0

            # ---------- LCD ----------

            lcd.update(
                mode,
                status,
                current_brightness,
                current_cct_mode
            )

            # ---------- DISPLAY ----------

            current_blinks = (
                total_blinks - blink_start_count
                if measurement_started
                else 0
            )

            y = 30

            y = draw_text(
                frame,
                f"EAR : {ear:.3f}",
                y
            )

            y = draw_text(
                frame,
                f"Blinks : {current_blinks}",
                y
            )

            y = draw_text(
                frame,
                f"Mean BD : {mean_bd:.1f} ms",
                y
            )

            y = draw_text(
                frame,
                f"Current BD : {current_bd:.0f} ms",
                y
            )

            y = draw_text(
                frame,
                f"Eye Reduction : {eor:.1f}%",
                y
            )

            y = draw_text(
                frame,
                f"Mean EOR : {mean_eor:.1f}%",
                y
            )

            y = draw_text(
                frame,
                f"Timer : {remaining:.0f} sec",
                y
            )

            y = draw_text(
                frame,
                f"Status : {status}",
                y,
                (0, 0, 255)
            )

            if fatigue_level is not None:

                y = draw_text(
                    frame,
                    f"Score : {fatigue_score:.2f}",
                    y
                )

                y = draw_text(
                    frame,
                    f"BR:{br_level} "
                    f"BD:{bd_level} "
                    f"EOR:{eor_level}",
                    y
                )

            y = draw_text(
                frame,
                f"Mode : {mode}",
                y
            )

            y = draw_text(
                frame,
                f"Brightness : "
                f"{current_brightness}%",
                y
            )

            y = draw_text(
                frame,
                f"CCT : {current_cct_mode}",
                y
            )

            y = draw_text(
                frame,
                f"Lux : {lux}",
                y
            )

            draw_text(
                frame,
                f"Color Temp : {cct} K",
                y
            )

            cv2.imshow(
                "Eye Fatigue Detection",
                frame
            )

            if (
                cv2.waitKey(1) & 0xFF
            ) in (
                ord("q"),
                ord("Q")
            ):
                break

    except KeyboardInterrupt:
        print("[SYSTEM] Ctrl+C")

    except Exception as error:

        print(
            f"[ERROR] "
            f"{type(error).__name__}: "
            f"{error}"
        )

        traceback.print_exc()

    finally:

        print("[SYSTEM] Cleaning up...")

        cleanup_items = [
            (camera, "release"),
            (blink_detector, "close"),
            (eye_detector, "close"),
            (lcd, "cleanup"),
            (led, "cleanup"),
            (als, "close"),
            (buttons, "cleanup")
        ]

        for obj, method in cleanup_items:

            if (
                obj is not None
                and hasattr(obj, method)
            ):
                try:
                    getattr(obj, method)()
                except Exception:
                    pass

        cv2.destroyAllWindows()

        print("[SYSTEM] Program ended")


if __name__ == "__main__":
    main()
