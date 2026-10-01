import cv2
import mediapipe as mp
import time
from scipy.spatial import distance

from utils.config import *


class BlinkDetector:

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self):

        # 전체 누적 Blink
        self.total_blinks = 0

        # 최근 1분 Blink
        self.minute_blinks = 0

        # Blink Rate (1분마다 갱신)
        self.blink_rate = 0

        self.counter = 0
        self.eye_closed = False

        self.start_time = time.time()

        mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def eye_aspect_ratio(self, eye):

        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])

        return (A + B) / (2.0 * C)

    def update(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        ear = 0

        if results.multi_face_landmarks:

            face = results.multi_face_landmarks[0]

            h, w, _ = frame.shape

            left_eye = []
            right_eye = []

            # -----------------------------
            # Left Eye
            # -----------------------------
            for idx in self.LEFT_EYE:

                lm = face.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                left_eye.append((x, y))

                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            # -----------------------------
            # Right Eye
            # -----------------------------
            for idx in self.RIGHT_EYE:

                lm = face.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                right_eye.append((x, y))

                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            leftEAR = self.eye_aspect_ratio(left_eye)
            rightEAR = self.eye_aspect_ratio(right_eye)

            ear = (leftEAR + rightEAR) / 2.0

            # -----------------------------
            # Blink Detection
            # -----------------------------
            if ear < EYE_AR_THRESH:

                self.counter += 1

                if self.counter >= EYE_AR_CONSEC_FRAMES:
                    self.eye_closed = True

            else:

                if self.eye_closed:

                    # 전체 누적
                    self.total_blinks += 1

                    # 최근 1분
                    self.minute_blinks += 1

                self.counter = 0
                self.eye_closed = False

            # -----------------------------
            # Blink Rate (1분마다 계산)
            # -----------------------------
            current_time = time.time()

            if current_time - self.start_time >= BLINK_RATE_INTERVAL:

                self.blink_rate = self.minute_blinks

                # 다음 1분 측정을 위해 초기화
                self.minute_blinks = 0

                self.start_time = current_time

        return {
            "frame": frame,
            "ear": ear,
            "total_blinks": self.total_blinks,
            "minute_blinks": self.minute_blinks,
            "blink_rate": self.blink_rate
        }