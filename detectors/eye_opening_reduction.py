import cv2
import mediapipe as mp
import time
import numpy as np

from collections import deque
from scipy.spatial import distance

from utils.config import EYE_AR_THRESH


class EyeOpeningReduction:

    # =========================================================
    # Eye Landmark
    # =========================================================

    LEFT_EYE = [
        33, 160, 158,
        133, 153, 144
    ]

    RIGHT_EYE = [
        362, 385, 387,
        263, 373, 380
    ]

    def __init__(self):

        # =====================================================
        # Calibration 설정
        # =====================================================

        # 처음 5초 동안 baseline 측정
        self.calibration_time = 5.0

        self.calibration_start = None

        self.baseline_samples = []

        self.baseline_eye_size = None

        self.calibrated = False

        # =====================================================
        # 현재 눈 크기 / 감소율
        # =====================================================

        self.current_eye_size = 0.0

        self.reduction_ratio = 0.0

        # =====================================================
        # 흔들림 완화
        # 최근 15프레임 평균
        # =====================================================

        self.recent_eye_sizes = deque(
            maxlen=15
        )

        # =====================================================
        # MediaPipe FaceMesh
        # =====================================================

        mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # =========================================================
    # EAR 계산
    #
    # Blink 상태를 제외하기 위한 용도
    # =========================================================

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

        ear = (
            A + B
        ) / (
            2.0 * C
        )

        return ear

    # =========================================================
    # 눈 뜬 크기 계산
    #
    # 눈 면적 / 눈 가로길이^2
    #
    # 거리 변화 영향을 줄이기 위해 정규화
    # =========================================================

    def calculate_eye_size(self, eye):

        eye_points = np.array(
            eye,
            dtype=np.float32
        )

        # 눈 면적
        eye_area = cv2.contourArea(
            eye_points
        )

        # 눈 가로 길이
        eye_width = distance.euclidean(
            eye[0],
            eye[3]
        )

        if eye_width <= 0:

            return 0.0

        eye_size = (
            eye_area
            /
            (eye_width ** 2)
        )

        return eye_size

    # =========================================================
    # Update
    # =========================================================

    def update(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(
            rgb
        )

        # =====================================================
        # 얼굴 미검출
        # =====================================================

        if not results.multi_face_landmarks:

            return {
                "face_detected": False,
                "calibrated": self.calibrated,
                "baseline_eye_size": self.baseline_eye_size,
                "current_eye_size": self.current_eye_size,
                "reduction_ratio": self.reduction_ratio
            }

        # =====================================================
        # Face Landmark
        # =====================================================

        face = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        left_eye = []
        right_eye = []

        # =====================================================
        # Left Eye
        # =====================================================

        for idx in self.LEFT_EYE:

            lm = face.landmark[idx]

            x = int(
                lm.x * w
            )

            y = int(
                lm.y * h
            )

            left_eye.append(
                (x, y)
            )

        # =====================================================
        # Right Eye
        # =====================================================

        for idx in self.RIGHT_EYE:

            lm = face.landmark[idx]

            x = int(
                lm.x * w
            )

            y = int(
                lm.y * h
            )

            right_eye.append(
                (x, y)
            )

        # =====================================================
        # EAR
        # =====================================================

        left_ear = self.eye_aspect_ratio(
            left_eye
        )

        right_ear = self.eye_aspect_ratio(
            right_eye
        )

        ear = (
            left_ear
            +
            right_ear
        ) / 2.0

        # =====================================================
        # 눈 크기
        # =====================================================

        left_eye_size = (
            self.calculate_eye_size(
                left_eye
            )
        )

        right_eye_size = (
            self.calculate_eye_size(
                right_eye
            )
        )

        eye_size = (
            left_eye_size
            +
            right_eye_size
        ) / 2.0

        current_time = time.time()

        # =====================================================
        # Blink 상태는 제외
        #
        # 눈을 감는 순간 계산하면 감소율이
        # 80~100%까지 튈 수 있으므로 제외
        # =====================================================

        if ear > EYE_AR_THRESH:

            # =================================================
            # Calibration 시작
            # =================================================

            if self.calibration_start is None:

                self.calibration_start = (
                    current_time
                )

                print()
                print(
                    "========================================"
                )
                print(
                    "Eye Opening Calibration Start"
                )
                print(
                    "Keep your eyes open naturally."
                )
                print(
                    "Calibration Time : 5 sec"
                )
                print(
                    "========================================"
                )

            # =================================================
            # 처음 5초 Calibration
            # =================================================

            if not self.calibrated:

                elapsed = (
                    current_time
                    -
                    self.calibration_start
                )

                if elapsed < self.calibration_time:

                    if eye_size > 0:

                        self.baseline_samples.append(
                            eye_size
                        )

                    self.current_eye_size = (
                        eye_size
                    )

                else:

                    if len(
                        self.baseline_samples
                    ) > 0:

                        # =====================================
                        # Baseline = 처음 5초 평균
                        # =====================================

                        self.baseline_eye_size = float(
                            np.mean(
                                self.baseline_samples
                            )
                        )

                        self.current_eye_size = (
                            self.baseline_eye_size
                        )

                        self.reduction_ratio = 0.0

                        self.calibrated = True

                        # Moving average 초기화
                        self.recent_eye_sizes.clear()

                        for _ in range(15):

                            self.recent_eye_sizes.append(
                                self.baseline_eye_size
                            )

                        print()
                        print(
                            "========================================"
                        )
                        print(
                            "Eye Opening Calibration Complete"
                        )

                        print(
                            "Baseline Eye Size : "
                            f"{self.baseline_eye_size:.4f}"
                        )

                        print(
                            "Eye Opening Reduction : "
                            "0.0%"
                        )

                        print(
                            "========================================"
                        )

            # =================================================
            # Calibration 완료 후
            # =================================================

            else:

                if eye_size > 0:

                    # 최근 15프레임 저장
                    self.recent_eye_sizes.append(
                        eye_size
                    )

                    # 최근 15프레임 평균
                    self.current_eye_size = float(
                        np.mean(
                            self.recent_eye_sizes
                        )
                    )

                    # =========================================
                    # Eye Opening Reduction
                    #
                    # (Baseline - Current)
                    # -------------------- × 100
                    #       Baseline
                    # =========================================

                    if (
                        self.baseline_eye_size
                        is not None
                        and
                        self.baseline_eye_size > 0
                    ):

                        reduction = (
                            (
                                self.baseline_eye_size
                                -
                                self.current_eye_size
                            )
                            /
                            self.baseline_eye_size
                        ) * 100.0

                        # Baseline보다 눈을 더 크게 뜬 경우
                        if reduction < 0:

                            reduction = 0.0

                        # 최대 100%
                        if reduction > 100:

                            reduction = 100.0

                        self.reduction_ratio = float(
                            reduction
                        )

        # =====================================================
        # 결과
        # =====================================================

        return {

            "face_detected":
                True,

            "calibrated":
                self.calibrated,

            "baseline_eye_size":
                self.baseline_eye_size,

            "current_eye_size":
                self.current_eye_size,

            "reduction_ratio":
                self.reduction_ratio
        }

    # =========================================================
    # Cleanup
    # =========================================================

    def close(self):

        if self.face_mesh is not None:

            self.face_mesh.close()
