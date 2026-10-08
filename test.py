import cv2
import time
import traceback

from camera.camera import Camera

from detectors.blink_detector import BlinkDetector
from detectors.blink_duration import BlinkDuration
from detectors.eye_opening_reduction import EyeOpeningReduction

from utils.config import EYE_AR_THRESH


TEST_DURATION = 60.0


def draw_text(
    frame,
    text,
    y,
    color=(255, 255, 255),
    scale=0.65
):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2
    )

    return y + 32


def main():

    print("========================================")
    print("     Eye Fatigue 1-Minute Test")
    print("========================================")

    camera = None
    blink_detector = None
    blink_duration = None
    eye_opening_reduction = None

    try:

        # =====================================================
        # Camera
        # =====================================================

        print("[INIT] Camera...")

        camera = Camera()

        # =====================================================
        # Eye Opening Reduction
        #
        # 먼저 5초 개인 눈 크기 calibration 수행
        # =====================================================

        print("[INIT] EyeOpeningReduction...")

        eye_opening_reduction = (
            EyeOpeningReduction()
        )

        print()
        print(
            "Eye Opening Calibration을 시작합니다."
        )
        print(
            "카메라를 자연스럽게 바라봐 주세요."
        )

        # =====================================================
        # Calibration
        # =====================================================

        while True:

            frame = camera.read()

            if frame is None:

                print(
                    "[ERROR] Camera frame is None"
                )

                return

            frame = cv2.flip(
                frame,
                1
            )

            eye_result = (
                eye_opening_reduction.update(
                    frame.copy()
                )
            )

            calibrated = (
                eye_result.get(
                    "calibrated",
                    False
                )
            )

            baseline_eye_size = (
                eye_result.get(
                    "baseline_eye_size",
                    None
                )
            )

            # 화면
            cv2.putText(
                frame,
                "Eye Opening Calibration...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "1-Minute Eye Test",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                ord("Q")
            ):

                return

            if calibrated:

                break

        print()
        print("========================================")
        print("Calibration Complete")

        if baseline_eye_size is not None:

            print(
                f"Baseline Eye Size : "
                f"{baseline_eye_size:.4f}"
            )

        print("========================================")

        # =====================================================
        # Blink 관련 객체를 Calibration 이후 생성
        #
        # 이렇게 해야 정확히 새로운 60초 측정이 시작됨
        # =====================================================

        print("[INIT] BlinkDetector...")

        blink_detector = (
            BlinkDetector()
        )

        print("[INIT] BlinkDuration...")

        blink_duration = (
            BlinkDuration(
                close_threshold=0.21,
                open_threshold=0.24
            )
        )

        # =====================================================
        # 측정 변수
        # =====================================================

        test_start_time = time.time()

        # Eye Opening Reduction 값 저장
        eye_reduction_samples = []

        # 시작 시 Blink 총 개수
        initial_total_blinks = 0

        last_terminal_print = 0.0

        print()
        print("========================================")
        print("      60 SECOND TEST START")
        print("========================================")

        # =====================================================
        # 60초 측정
        # =====================================================

        while True:

            now = time.time()

            elapsed = (
                now
                -
                test_start_time
            )

            remaining = max(
                0.0,
                TEST_DURATION
                -
                elapsed
            )

            if elapsed >= TEST_DURATION:

                break

            # =================================================
            # Camera
            # =================================================

            frame = camera.read()

            if frame is None:

                print(
                    "[ERROR] Camera frame is None"
                )

                break

            frame = cv2.flip(
                frame,
                1
            )

            eye_frame = frame.copy()

            # =================================================
            # Blink Detector
            # =================================================

            blink_result = (
                blink_detector.update(
                    frame
                )
            )

            frame = blink_result.get(
                "frame",
                frame
            )

            ear = blink_result.get(
                "ear",
                0.0
            )

            total_blinks = (
                blink_result.get(
                    "total_blinks",
                    0
                )
            )

            minute_blinks = (
                blink_result.get(
                    "minute_blinks",
                    0
                )
            )

            # =================================================
            # Blink Duration
            # =================================================

            duration_result = (
                blink_duration.update(
                    ear
                )
            )

            in_blink = (
                duration_result.get(
                    "in_blink",
                    False
                )
            )

            current_blink_duration = (
                duration_result.get(
                    "current_blink_duration_ms",
                    0.0
                )
            )

            mean_blink_duration = (
                duration_result.get(
                    "mean_blink_duration_ms",
                    0.0
                )
            )

            # =================================================
            # Eye Opening Reduction
            # =================================================

            eye_result = (
                eye_opening_reduction.update(
                    eye_frame
                )
            )

            face_detected = (
                eye_result.get(
                    "face_detected",
                    False
                )
            )

            eye_reduction_ratio = (
                eye_result.get(
                    "reduction_ratio",
                    0.0
                )
            )

            # =================================================
            # Eye Opening Reduction 평균값 저장
            #
            # 얼굴이 검출되고
            # 눈을 감는 Blink 순간이 아닐 때만 저장
            # =================================================

            if (
                face_detected
                and
                ear > EYE_AR_THRESH
            ):

                eye_reduction_samples.append(
                    eye_reduction_ratio
                )

            # =================================================
            # 현재 테스트 Blink Rate
            #
            # 아직 1분이 끝나기 전이므로
            # 현재 누적 blink 개수 표시
            # =================================================

            current_blink_count = (
                total_blinks
                -
                initial_total_blinks
            )

            # =================================================
            # 현재 Eye Opening Reduction 평균
            # =================================================

            if len(
                eye_reduction_samples
            ) > 0:

                current_mean_reduction = (
                    sum(
                        eye_reduction_samples
                    )
                    /
                    len(
                        eye_reduction_samples
                    )
                )

            else:

                current_mean_reduction = 0.0

            # =================================================
            # Terminal
            # 1초마다 출력
            # =================================================

            if (
                now
                -
                last_terminal_print
                >=
                1.0
            ):

                print(
                    f"[TEST] "
                    f"Time={elapsed:.0f}s | "
                    f"Blinks={current_blink_count} | "
                    f"Mean BD="
                    f"{mean_blink_duration:.1f}ms | "
                    f"Eye Reduction="
                    f"{current_mean_reduction:.1f}%"
                )

                last_terminal_print = now

            # =================================================
            # Camera 화면
            # =================================================

            y = 35

            y = draw_text(
                frame,
                "1-Minute Eye Fatigue Test",
                y,
                (0, 255, 255),
                0.70
            )

            y = draw_text(
                frame,
                f"Time Left : "
                f"{remaining:.0f} sec",
                y
            )

            y = draw_text(
                frame,
                f"EAR : "
                f"{ear:.3f}",
                y,
                (0, 255, 0)
            )

            y = draw_text(
                frame,
                f"Blink Count : "
                f"{current_blink_count}",
                y,
                (255, 255, 0)
            )

            y = draw_text(
                frame,
                (
                    "Blink : DETECTING"
                    if in_blink
                    else
                    "Blink : OPEN"
                ),
                y
            )

            y = draw_text(
                frame,
                f"Current BD : "
                f"{current_blink_duration:.0f} ms",
                y
            )

            y = draw_text(
                frame,
                f"Mean BD : "
                f"{mean_blink_duration:.1f} ms",
                y,
                (0, 255, 255)
            )

            y = draw_text(
                frame,
                f"Current Eye Reduction : "
                f"{eye_reduction_ratio:.1f}%",
                y
            )

            y = draw_text(
                frame,
                f"Mean Eye Reduction : "
                f"{current_mean_reduction:.1f}%",
                y,
                (0, 255, 255)
            )

            cv2.imshow(
                "1-Minute Eye Test",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                ord("Q")
            ):

                print(
                    "[SYSTEM] Test cancelled"
                )

                return

        # =====================================================
        # 최종 결과
        # =====================================================

        final_total_blinks = (
            blink_result.get(
                "total_blinks",
                0
            )
        )

        # 60초이므로 Blink Count 자체가
        # Blink Rate (/min)
        blink_rate = (
            final_total_blinks
            -
            initial_total_blinks
        )

        # =====================================================
        # Blink Duration 최종값
        #
        # BlinkDuration 내부에 60초 결과가 저장되어 있음
        # =====================================================

        duration_result = (
            blink_duration.update(
                ear
            )
        )

        mean_blink_duration = (
            duration_result.get(
                "mean_blink_duration_ms",
                0.0
            )
        )

        duration_blink_count = (
            duration_result.get(
                "blink_count",
                0
            )
        )

        # =====================================================
        # Eye Opening Reduction 최종 평균
        # =====================================================

        if len(
            eye_reduction_samples
        ) > 0:

            mean_eye_reduction = (
                sum(
                    eye_reduction_samples
                )
                /
                len(
                    eye_reduction_samples
                )
            )

        else:

            mean_eye_reduction = 0.0

        # =====================================================
        # 결과 출력
        # =====================================================

        print()
        print("========================================")
        print("       1-MINUTE TEST RESULT")
        print("========================================")

        print(
            f"Blink Rate                : "
            f"{blink_rate} /min"
        )

        print(
            f"Blink Duration Count      : "
            f"{duration_blink_count}"
        )

        print(
            f"Mean Blink Duration       : "
            f"{mean_blink_duration:.1f} ms"
        )

        print(
            f"Mean Eye Opening Reduction: "
            f"{mean_eye_reduction:.1f} %"
        )

        print("========================================")

        # =====================================================
        # 결과 화면 유지
        # =====================================================

        result_frame = frame.copy()

        overlay = result_frame.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (620, 230),
            (0, 0, 0),
            -1
        )

        result_frame = cv2.addWeighted(
            overlay,
            0.6,
            result_frame,
            0.4,
            0
        )

        y = 45

        y = draw_text(
            result_frame,
            "1-Minute Test Complete",
            y,
            (0, 255, 255),
            0.75
        )

        y = draw_text(
            result_frame,
            f"Blink Rate : "
            f"{blink_rate}/min",
            y
        )

        y = draw_text(
            result_frame,
            f"Mean Blink Duration : "
            f"{mean_blink_duration:.1f} ms",
            y
        )

        y = draw_text(
            result_frame,
            f"Mean Eye Reduction : "
            f"{mean_eye_reduction:.1f}%",
            y
        )

        y = draw_text(
            result_frame,
            "Press Q to exit",
            y
        )

        while True:

            cv2.imshow(
                "1-Minute Eye Test",
                result_frame
            )

            key = cv2.waitKey(20) & 0xFF

            if key in (
                ord("q"),
                ord("Q")
            ):

                break

    except KeyboardInterrupt:

        print(
            "\n[SYSTEM] Ctrl+C"
        )

    except Exception as error:

        print(
            f"[ERROR] "
            f"{type(error).__name__}: "
            f"{error}"
        )

        traceback.print_exc()

    finally:

        print(
            "[SYSTEM] Cleaning up..."
        )

        if camera is not None:

            try:
                camera.release()
            except Exception:
                pass

        if blink_detector is not None:

            try:
                blink_detector.close()
            except Exception:
                pass

        if eye_opening_reduction is not None:

            try:
                eye_opening_reduction.close()
            except Exception:
                pass

        cv2.destroyAllWindows()

        print(
            "[SYSTEM] Test ended"
        )


if __name__ == "__main__":

    main()
