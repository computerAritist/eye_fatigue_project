import cv2

from camera.camera import Camera
from controllers.led_controller import LEDController
from detectors.blink_detector import BlinkDetector
from detectors.fatigue_detector import FatigueDetector
from sensors.als_sensor import ALSSensor


def main():

    # -----------------------------
    # 객체 생성
    # -----------------------------
    camera = Camera()
    blink_detector = BlinkDetector()
    fatigue_detector = FatigueDetector()
    led = LEDController()
    als = ALSSensor()

    # -----------------------------
    # 시작과 동시에 기본 밝기(30%)
    # -----------------------------
    led.set_brightness(30)
    current_brightness = 30

    try:

        while True:

            # -----------------------------
            # 카메라 프레임 읽기
            # -----------------------------
            frame = camera.read()

            if frame is None:
                break

            frame = cv2.flip(frame, 1)

            # -----------------------------
            # Blink Detection
            # -----------------------------
            result = blink_detector.update(frame)
            frame = result["frame"]

            # -----------------------------
            # ALS Sensor
            # -----------------------------
            lux, cct = als.read()

            # -----------------------------
            # Fatigue Detection
            # -----------------------------
            status = fatigue_detector.update(result["blink_rate"])

            # -----------------------------
            # LED 밝기 결정
            # -----------------------------
            if status == "Measuring...":
                brightness = 10

            elif status == "NORMAL":
                brightness = 30

            elif status == "WARNING":
                brightness = 60

            elif status == "FATIGUE":
                brightness = 100

            else:
                brightness = 30

            # -----------------------------
            # LED 상태가 바뀔 때만 PWM 변경
            # -----------------------------
            if brightness != current_brightness:
                print(f"LED -> {brightness}%")
                led.set_brightness(brightness)
                current_brightness = brightness

            # -----------------------------
            # 화면 출력
            # -----------------------------
            cv2.putText(
                frame,
                f"EAR : {result['ear']:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Minute Blinks : {result['minute_blinks']}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Blink Rate : {result['blink_rate']}/min",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Status : {status}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Illuminance : {lux} lx",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Color Temp : {cct} K",
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow("Eye Fatigue Detection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

    finally:
        camera.release()
        led.cleanup()
        als.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()