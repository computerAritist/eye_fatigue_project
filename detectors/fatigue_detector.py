from utils.config import NORMAL_THRESHOLD, WARNING_THRESHOLD


class FatigueDetector:

    def __init__(self):
        self.status = "Measuring..."
        self.first_measurement = False

    def update(self, blink_rate):

        # 아직 첫 번째 측정이 끝나지 않은 경우
        if not self.first_measurement:

            if blink_rate == 0:
                return self.status

            self.first_measurement = True

        # Blink Rate에 따른 피로도 판단
        if blink_rate >= NORMAL_THRESHOLD:
            self.status = "NORMAL"

        elif blink_rate >= WARNING_THRESHOLD:
            self.status = "WARNING"

        else:
            self.status = "FATIGUE"

        return self.status