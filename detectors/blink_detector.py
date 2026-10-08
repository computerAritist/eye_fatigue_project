import cv2
import mediapipe as mp
import time

from scipy.spatial import distance

from utils.config import BLINK_RATE_INTERVAL


class BlinkDetector:

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self):

        # ============================================
        # EAR 기준
        # ============================================

        self.close_threshold = 0.20
        self.open_threshold = 0.23

        # 너무 짧은 노이즈 제거
        self.min_blink_duration = 0.06

        # 너무 긴 눈 감김 제외
        self.max_blink_duration = 1.0

        # blink 중 EAR이 이 값 아래까지는 내려가야 인정
        self.required_min_ear = 0.18

        # ============================================
        # Blink 상태
        # ============================================

        self.in_blink = False

        self.blink_start_time = None

        self.min_ear_during_blink = 1.0

        # ============================================
        # Count
        # ============================================

        self.total_blinks = 0
        self.minute_blinks = 0
        self.blink_rate = 0

        self.start_time = time.time()

        # ============================================
        # MediaPipe
        # ============================================

        mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def eye_aspect_ratio(self, eye):

        A = distance.euclidean(
            eye[1],
            eye[5]
        )

        B = distance.euclidean(
            eye[2],
            eye[4]
        )

        C = distance.euclidean(
            eye[0],
            eye[3]
        )

        if C <= 0:
            return 0.0

        return (A + B) / (2.0 * C)

    def update(self, frame):

        now = time.time()

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(
            rgb
        )

        ear = 0.0
        face_detected = False

        # ============================================
        # 얼굴 검출
        # ============================================

        if results.multi_face_landmarks:

            face_detected = True

            face = results.multi_face_landmarks[0]

            h, w, _ = frame.shape

            left_eye = []
            right_eye = []

            for idx in self.LEFT_EYE:

                lm = face.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                left_eye.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 0, 255),
                    -1
                )

            for idx in self.RIGHT_EYE:

                lm = face.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                right_eye.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 0, 255),
                    -1
                )

            left_ear = self.eye_aspect_ratio(
                left_eye
            )

            right_ear = self.eye_aspect_ratio(
                right_eye
            )

            ear = (
                left_ear + right_ear
            ) / 2.0

            # ========================================
            # Blink 시작
            # ========================================

            if not self.in_blink:

                if ear < self.close_threshold:

                    self.in_blink = True

                    self.blink_start_time = now

                    self.min_ear_during_blink = ear

            # ========================================
            # Blink 진행 중
            # ========================================

            else:

                if ear < self.min_ear_during_blink:

                    self.min_ear_during_blink = ear

                # ====================================
                # 다시 눈이 열림
                # ====================================

                if ear >= self.open_threshold:

                    duration = (
                        now -
                        self.blink_start_time
                    )

                    valid_duration = (
                        self.min_blink_duration
                        <= duration
                        <= self.max_blink_duration
                    )

                    valid_ear_drop = (
                        self.min_ear_during_blink
                        <= self.required_min_ear
                    )

                    # =================================
                    # 정상 Blink만 Count
                    # =================================

                    if (
                        valid_duration
                        and
                        valid_ear_drop
                    ):

                        self.total_blinks += 1

                        self.minute_blinks += 1

                        print(
                            f"[BLINK] "
                            f"Total={self.total_blinks} | "
                            f"Minute={self.minute_blinks} | "
                            f"Duration={duration * 1000:.0f} ms | "
                            f"Min EAR={self.min_ear_during_blink:.3f}"
                        )

                    else:

                        print(
                            f"[REJECT] "
                            f"Duration={duration * 1000:.0f} ms | "
                            f"Min EAR={self.min_ear_during_blink:.3f}"
                        )

                    # 다음 Blink 준비
                    self.in_blink = False

                    self.blink_start_time = None

                    self.min_ear_during_blink = 1.0

        # ============================================
        # 얼굴 미검출
        # ============================================

        else:

            self.in_blink = False

            self.blink_start_time = None

            self.min_ear_during_blink = 1.0

        # ============================================
        # 1분 Blink Rate
        # ============================================

        elapsed = (
            now -
            self.start_time
        )

        if elapsed >= BLINK_RATE_INTERVAL:

            self.blink_rate = (
                self.minute_blinks
            )

            print()
            print("==============================")
            print(
                f"Blink Rate : "
                f"{self.blink_rate}/min"
            )
            print("==============================")

            self.minute_blinks = 0

            self.start_time = now

        # ============================================
        # 결과
        # ============================================

        return {

            "frame": frame,

            "face_detected":
                face_detected,

            "ear":
                ear,

            "total_blinks":
                self.total_blinks,

            "minute_blinks":
                self.minute_blinks,

            "blink_rate":
                self.blink_rate,

            "eye_closed":
                self.in_blink
        }

    def close(self):

        if self.face_mesh is not None:

            self.face_mesh.close()
