import time


class BlinkDuration:

    def __init__(
        self,
        close_threshold=0.21,
        open_threshold=0.24
    ):

        # =====================================================
        # 1분 측정
        # =====================================================

        self.measurement_interval = 60.0
        self.measurement_start = time.time()

        # =====================================================
        # Blink 시작 / 종료 Threshold
        #
        # close_threshold:
        # 눈이 감기기 시작했다고 판단하는 EAR
        #
        # open_threshold:
        # 눈이 다시 충분히 열렸다고 판단하는 EAR
        #
        # 두 값을 다르게 둬서 threshold 근처의 흔들림 방지
        # =====================================================

        self.close_threshold = close_threshold
        self.open_threshold = open_threshold

        # =====================================================
        # Blink 상태
        # =====================================================

        self.in_blink = False
        self.blink_start_time = None

        # 현재 진행 중인 Blink Duration
        self.current_blink_duration_ms = 0.0

        # =====================================================
        # 1분 동안 완료된 Blink Duration 저장
        # =====================================================

        self.blink_durations = []

        # 최근 1분 결과
        self.mean_blink_duration_ms = 0.0
        self.blink_count = 0

        self.measurement_complete = False

    # =========================================================
    # Update
    # =========================================================

    def update(self, ear):

        now = time.time()

        self.measurement_complete = False

        # =====================================================
        # EAR=0
        # 얼굴 미검출 가능성이 있으므로 측정 제외
        # =====================================================

        if ear <= 0:

            self.current_blink_duration_ms = 0.0

            return self._make_result(now)

        # =====================================================
        # Blink 시작
        #
        # 정상 눈뜬 상태에서 EAR이 threshold 아래로 내려감
        # =====================================================

        if (
            not self.in_blink
            and
            ear < self.close_threshold
        ):

            self.in_blink = True

            self.blink_start_time = now

            self.current_blink_duration_ms = 0.0

        # =====================================================
        # Blink 진행 중
        # =====================================================

        if (
            self.in_blink
            and
            self.blink_start_time is not None
        ):

            self.current_blink_duration_ms = (
                now
                -
                self.blink_start_time
            ) * 1000.0

        # =====================================================
        # Blink 종료
        #
        # 다시 충분히 눈을 뜨면 종료
        # =====================================================

        if (
            self.in_blink
            and
            ear >= self.open_threshold
            and
            self.blink_start_time is not None
        ):

            duration_ms = (
                now
                -
                self.blink_start_time
            ) * 1000.0

            # =================================================
            # 이상치 제거
            #
            # 너무 짧은 값 = landmark noise 가능성
            # 너무 긴 값 = 단순 blink가 아니라 장시간 눈감음 가능성
            #
            # 초기값이며 실제 데이터 보고 조정 가능
            # =================================================

            if (
                duration_ms >= 80.0
                and
                duration_ms <= 2000.0
            ):

                self.blink_durations.append(
                    duration_ms
                )

            self.in_blink = False
            self.blink_start_time = None
            self.current_blink_duration_ms = 0.0

        # =====================================================
        # 결과
        # =====================================================

        return self._make_result(now)

    # =========================================================
    # 1분 평균 계산
    # =========================================================

    def _make_result(self, now):

        elapsed = (
            now
            -
            self.measurement_start
        )

        # =====================================================
        # 1분 완료
        # =====================================================

        if elapsed >= self.measurement_interval:

            self.blink_count = len(
                self.blink_durations
            )

            if self.blink_count > 0:

                self.mean_blink_duration_ms = (
                    sum(
                        self.blink_durations
                    )
                    /
                    self.blink_count
                )

            else:

                self.mean_blink_duration_ms = 0.0

            self.measurement_complete = True

            print()
            print(
                "========================================"
            )
            print(
                "Blink Duration - 1 Minute Result"
            )

            print(
                f"Blink Count : "
                f"{self.blink_count}"
            )

            print(
                f"Mean Blink Duration : "
                f"{self.mean_blink_duration_ms:.1f} ms"
            )

            print(
                "========================================"
            )

            # 다음 1분 측정
            self.blink_durations = []

            self.measurement_start = now

            elapsed = 0.0

        remaining_time = max(
            0.0,
            self.measurement_interval
            -
            elapsed
        )

        return {

            "in_blink":
                self.in_blink,

            "current_blink_duration_ms":
                self.current_blink_duration_ms,

            "mean_blink_duration_ms":
                self.mean_blink_duration_ms,

            "blink_count":
                self.blink_count,

            "remaining_time":
                remaining_time,

            "measurement_complete":
                self.measurement_complete
        }
