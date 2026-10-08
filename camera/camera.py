import cv2
import time

from utils.config import CAMERA_INDEX


class Camera:

    def __init__(self):

        self.cap = None
        self.first_frame = None

        print(
            f"[CAMERA] Opening camera index "
            f"{CAMERA_INDEX}"
        )

        # =====================================================
        # 1차 시도
        # OpenCV 기본 방식
        # =====================================================

        self.cap = cv2.VideoCapture(
            CAMERA_INDEX
        )

        if self.cap.isOpened():

            self.cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                640
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                480
            )

            time.sleep(0.5)

            ret, frame = self.cap.read()

            if ret and frame is not None:

                self.first_frame = frame

                print(
                    "[CAMERA] Default backend OK"
                )

                print(
                    f"[CAMERA] "
                    f"{frame.shape[1]}x"
                    f"{frame.shape[0]}"
                )

                return

        # =====================================================
        # 1차 실패
        # =====================================================

        if self.cap is not None:

            self.cap.release()

        print(
            "[CAMERA] Default backend failed"
        )

        print(
            "[CAMERA] Trying V4L2 + MJPG..."
        )

        # =====================================================
        # 2차 시도
        # V4L2 + MJPG
        # =====================================================

        self.cap = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_V4L2
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                "카메라를 열 수 없습니다."
            )

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(
                "M",
                "J",
                "P",
                "G"
            )
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            30
        )

        time.sleep(0.5)

        ret, frame = self.cap.read()

        if not ret or frame is None:

            self.cap.release()

            raise RuntimeError(
                "카메라는 열렸지만 "
                "프레임을 읽을 수 없습니다."
            )

        self.first_frame = frame

        print(
            "[CAMERA] V4L2 + MJPG OK"
        )

        print(
            f"[CAMERA] "
            f"{frame.shape[1]}x"
            f"{frame.shape[0]}"
        )

    # =========================================================
    # Frame Read
    # =========================================================

    def read(self):

        # 처음 성공했던 프레임 먼저 반환
        if self.first_frame is not None:

            frame = self.first_frame

            self.first_frame = None

            return frame

        # 최대 3번 재시도
        for _ in range(3):

            ret, frame = self.cap.read()

            if (
                ret
                and
                frame is not None
            ):

                return frame

            time.sleep(0.05)

        return None

    # =========================================================
    # Release
    # =========================================================

    def release(self):

        if self.cap is not None:

            if self.cap.isOpened():

                self.cap.release()

        print(
            "[CAMERA] Released"
        )
