import cv2
from utils.config import CAMERA_INDEX


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():
            raise Exception("카메라를 열 수 없습니다.")

    def read(self):
        """
        카메라에서 프레임을 읽는다.
        return: frame
        """
        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def release(self):
        """
        카메라 종료
        """
        self.cap.release()
        cv2.destroyAllWindows()