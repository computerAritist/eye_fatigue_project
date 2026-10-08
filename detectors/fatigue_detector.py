class FatigueDetector:

    def __init__(self):

        # ============================================
        # 가중치
        # ============================================

        self.weight_br = 0.30
        self.weight_bd = 0.45
        self.weight_eor = 0.25

        # ============================================
        # 현재 결과
        # ============================================

        self.fatigue_score = 0.0
        self.fatigue_level = None

        self.br_level = None
        self.bd_level = None
        self.eor_level = None

        self.status = "Measuring..."

    # ================================================
    # Blink Rate -> 1~5단계
    # ================================================

    @staticmethod
    def get_br_level(blink_rate):

        blink_rate = float(blink_rate)

        # 20~30 / min
        if 20 <= blink_rate <= 30:
            return 1

        # 17~19 또는 30 초과
        elif blink_rate > 30:
            return 2

        elif 17 <= blink_rate < 20:
            return 2

        # 14~16
        elif 14 <= blink_rate < 17:
            return 3

        # 10~13
        elif 10 <= blink_rate < 14:
            return 4

        # 9 이하
        else:
            return 5

    # ================================================
    # Mean Blink Duration -> 1~5단계
    # ================================================

    @staticmethod
    def get_bd_level(mean_blink_duration):

        value = float(mean_blink_duration)

        if value < 300:
            return 1

        elif value < 380:
            return 2

        elif value < 450:
            return 3

        elif value < 550:
            return 4

        else:
            return 5

    # ================================================
    # Eye Opening Reduction -> 1~5단계
    # ================================================

    @staticmethod
    def get_eor_level(eye_opening_reduction):

        value = float(eye_opening_reduction)

        if value < 1:
            return 1

        elif value < 2:
            return 2

        elif value < 3:
            return 3

        elif value < 5:
            return 4

        else:
            return 5

    # ================================================
    # 최종 점수 -> 피로도 단계
    # ================================================

    @staticmethod
    def score_to_level(score):

        if score < 1.50:
            return 1

        elif score < 2.50:
            return 2

        elif score < 3.50:
            return 3

        elif score < 4.50:
            return 4

        else:
            return 5

    # ================================================
    # 피로도 계산
    # ================================================

    def update(
        self,
        blink_rate,
        mean_blink_duration,
        eye_opening_reduction
    ):

        # 각각 1~5단계로 변환
        self.br_level = self.get_br_level(
            blink_rate
        )

        self.bd_level = self.get_bd_level(
            mean_blink_duration
        )

        self.eor_level = self.get_eor_level(
            eye_opening_reduction
        )

        # ============================================
        # 가중 평균
        #
        # BR  = 30%
        # BD  = 45%
        # EOR = 25%
        # ============================================

        self.fatigue_score = (
            self.weight_br
            * self.br_level

            +

            self.weight_bd
            * self.bd_level

            +

            self.weight_eor
            * self.eor_level
        )

        self.fatigue_level = (
            self.score_to_level(
                self.fatigue_score
            )
        )

        self.status = (
            f"LEVEL {self.fatigue_level}"
        )

        # ============================================
        # 터미널 출력
        # ============================================

        print()
        print("========================================")
        print("       FATIGUE RESULT")
        print("========================================")

        print(
            f"Blink Rate          : "
            f"{blink_rate}/min "
            f"-> Level {self.br_level}"
        )

        print(
            f"Mean Blink Duration : "
            f"{mean_blink_duration:.1f} ms "
            f"-> Level {self.bd_level}"
        )

        print(
            f"Eye Opening Red.    : "
            f"{eye_opening_reduction:.1f}% "
            f"-> Level {self.eor_level}"
        )

        print("----------------------------------------")

        print(
            f"Fatigue Score       : "
            f"{self.fatigue_score:.2f}"
        )

        print(
            f"Fatigue Level       : "
            f"{self.fatigue_level}"
        )

        print("========================================")
        print()

        return {

            "status":
                self.status,

            "level":
                self.fatigue_level,

            "score":
                self.fatigue_score,

            "br_level":
                self.br_level,

            "bd_level":
                self.bd_level,

            "eor_level":
                self.eor_level
        }
