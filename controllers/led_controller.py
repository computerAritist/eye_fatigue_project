# controllers/led_controller.py

import time
from pathlib import Path


class LEDController:
    """
    Raspberry Pi 5 Hardware PWM LED Controller

    GPIO18 -> PWM channel 2 -> White LED
    GPIO19 -> PWM channel 3 -> Yellow LED

    밝기:
        Lv.1 -> 20%
        Lv.2 -> 40%
        Lv.3 -> 60%
        Lv.4 -> 80%
        Lv.5 -> 100%

    색온도:
        Mode 1 -> White 5 : Yellow 0
        Mode 2 -> White 4 : Yellow 1
        Mode 3 -> White 3 : Yellow 2
        Mode 4 -> White 2 : Yellow 3
        Mode 5 -> White 1 : Yellow 4
        Mode 6 -> White 0 : Yellow 5
    """

    # ---------------------------------
    # PWM 설정
    # ---------------------------------

    PWMCHIP = 0

    WHITE_CHANNEL = 2
    YELLOW_CHANNEL = 3

    DEFAULT_FREQUENCY = 1000

    # ---------------------------------
    # 색온도 비율
    # ---------------------------------

    CCT_MODES = {
        1: (5, 0),
        2: (4, 1),
        3: (3, 2),
        4: (2, 3),
        5: (1, 4),
        6: (0, 5),
    }

    # ---------------------------------
    # 초기화
    # ---------------------------------

    def __init__(
        self,
        pwmchip=0,
        white_channel=2,
        yellow_channel=3,
        freq=1000,
        default_brightness=30,
        default_cct_mode=1
    ):

        self.freq = int(freq)

        if self.freq <= 0:
            raise ValueError(
                "PWM 주파수는 1Hz 이상이어야 합니다."
            )

        self.pwmchip = int(pwmchip)

        self.white_channel = int(white_channel)
        self.yellow_channel = int(yellow_channel)

        self.period_ns = int(
            1_000_000_000 / self.freq
        )

        self.chip_path = Path(
            f"/sys/class/pwm/pwmchip{self.pwmchip}"
        )

        self.white_pwm_path = (
            self.chip_path /
            f"pwm{self.white_channel}"
        )

        self.yellow_pwm_path = (
            self.chip_path /
            f"pwm{self.yellow_channel}"
        )

        self.white_exported_by_me = False
        self.yellow_exported_by_me = False

        self.enabled = False

        self.current_brightness = None
        self.current_cct_mode = None

        # ---------------------------------
        # PWM 초기화
        # ---------------------------------

        self._initialize_pwm()

        # ---------------------------------
        # 기본 상태
        # ---------------------------------

        self.current_brightness = self._limit_brightness(
            default_brightness
        )

        self.current_cct_mode = self._limit_cct_mode(
            default_cct_mode
        )

        self._apply_light()

        self.enabled = True

        print(
            "[LED] Hardware PWM started | "
            f"White=GPIO18(CH{self.white_channel}), "
            f"Yellow=GPIO19(CH{self.yellow_channel}), "
            f"frequency={self.freq}Hz"
        )

        print(
            "[LED] Initial state | "
            f"brightness={self.current_brightness}%, "
            f"CCT Mode={self.current_cct_mode}"
        )

    # =================================
    # PWM 초기화
    # =================================

    def _initialize_pwm(self):

        if not self.chip_path.exists():

            raise RuntimeError(
                f"PWM 장치를 찾을 수 없습니다: "
                f"{self.chip_path}\n"
                "config.txt에 "
                "'dtoverlay=pwm-2chan'이 있는지 확인하세요."
            )

        # ---------------------------------
        # White PWM
        # ---------------------------------

        if not self.white_pwm_path.exists():

            self._write(
                self.chip_path / "export",
                self.white_channel
            )

            self.white_exported_by_me = True

            self._wait_for_pwm(
                self.white_pwm_path
            )

        # ---------------------------------
        # Yellow PWM
        # ---------------------------------

        if not self.yellow_pwm_path.exists():

            self._write(
                self.chip_path / "export",
                self.yellow_channel
            )

            self.yellow_exported_by_me = True

            self._wait_for_pwm(
                self.yellow_pwm_path
            )

        # ---------------------------------
        # 최종 확인
        # ---------------------------------

        if not self.white_pwm_path.exists():

            raise RuntimeError(
                f"White LED PWM 채널을 열 수 없습니다: "
                f"{self.white_pwm_path}"
            )

        if not self.yellow_pwm_path.exists():

            raise RuntimeError(
                f"Yellow LED PWM 채널을 열 수 없습니다: "
                f"{self.yellow_pwm_path}"
            )

        # ---------------------------------
        # 두 PWM 정지
        # ---------------------------------

        self._safe_write(
            self.white_pwm_path / "enable",
            0
        )

        self._safe_write(
            self.yellow_pwm_path / "enable",
            0
        )

        # ---------------------------------
        # Duty 0으로 초기화
        # ---------------------------------

        self._write(
            self.white_pwm_path / "duty_cycle",
            0
        )

        self._write(
            self.yellow_pwm_path / "duty_cycle",
            0
        )

        # ---------------------------------
        # Period 설정
        # ---------------------------------

        self._write(
            self.white_pwm_path / "period",
            self.period_ns
        )

        self._write(
            self.yellow_pwm_path / "period",
            self.period_ns
        )

        # ---------------------------------
        # PWM 활성화
        # ---------------------------------

        self._write(
            self.white_pwm_path / "enable",
            1
        )

        self._write(
            self.yellow_pwm_path / "enable",
            1
        )

    # =================================
    # PWM 생성 대기
    # =================================

    @staticmethod
    def _wait_for_pwm(pwm_path):

        for _ in range(100):

            if pwm_path.exists():
                return

            time.sleep(0.01)

        raise RuntimeError(
            f"PWM 채널 생성에 실패했습니다: "
            f"{pwm_path}"
        )

    # =================================
    # 밝기 제한
    # =================================

    @staticmethod
    def _limit_brightness(brightness):

        return max(
            0,
            min(100, int(brightness))
        )

    # =================================
    # CCT Mode 제한
    # =================================

    @staticmethod
    def _limit_cct_mode(mode):

        mode = int(mode)

        if mode < 1:
            return 1

        if mode > 6:
            return 6

        return mode

    # =================================
    # 밝기 → Duty 계산
    # =================================

    def _brightness_to_duty(self, brightness):

        brightness = self._limit_brightness(
            brightness
        )

        return int(
            self.period_ns *
            brightness /
            100
        )

    # =================================
    # 실제 LED 출력 계산
    # =================================

    def _calculate_led_brightness(self):

        brightness = self.current_brightness
        cct_mode = self.current_cct_mode

        white_ratio, yellow_ratio = (
            self.CCT_MODES[cct_mode]
        )

        # White 0~100%
        white_brightness = (
            brightness *
            white_ratio /
            5
        )

        # Yellow 0~100%
        yellow_brightness = (
            brightness *
            yellow_ratio /
            5
        )

        return (
            white_brightness,
            yellow_brightness
        )

    # =================================
    # 실제 PWM 적용
    # =================================

    def _apply_light(self):

        white_brightness, yellow_brightness = (
            self._calculate_led_brightness()
        )

        white_duty = self._brightness_to_duty(
            white_brightness
        )

        yellow_duty = self._brightness_to_duty(
            yellow_brightness
        )

        # White LED
        self._write(
            self.white_pwm_path / "duty_cycle",
            white_duty
        )

        # Yellow LED
        self._write(
            self.yellow_pwm_path / "duty_cycle",
            yellow_duty
        )

        print(
            "[LED] Output | "
            f"CCT Mode={self.current_cct_mode}, "
            f"Brightness={self.current_brightness}% | "
            f"White={white_brightness:.1f}%, "
            f"Yellow={yellow_brightness:.1f}%"
        )

    # =================================
    # 밝기 설정
    # =================================

    def set_brightness(self, brightness):

        brightness = self._limit_brightness(
            brightness
        )

        if brightness == self.current_brightness:
            return

        self.current_brightness = brightness

        self._apply_light()

    # =================================
    # 색온도 설정
    # =================================

    def set_cct_mode(self, mode):

        mode = self._limit_cct_mode(mode)

        if mode == self.current_cct_mode:
            return

        self.current_cct_mode = mode

        self._apply_light()

    # =================================
    # 밝기 + 색온도 동시에 설정
    # =================================

    def set_light(self, brightness, cct_mode):

        brightness = self._limit_brightness(
            brightness
        )

        cct_mode = self._limit_cct_mode(
            cct_mode
        )

        self.current_brightness = brightness
        self.current_cct_mode = cct_mode

        self._apply_light()

    # =================================
    # 현재 상태 가져오기
    # =================================

    def get_brightness(self):

        return self.current_brightness

    def get_cct_mode(self):

        return self.current_cct_mode

    # =================================
    # LED ON
    # =================================

    def on(self):

        self.set_brightness(100)

    # =================================
    # LED OFF
    # =================================

    def off(self):

        self.set_brightness(0)

    # =================================
    # 파일 쓰기
    # =================================

    @staticmethod
    def _write(path, value):

        try:

            path.write_text(
                str(value),
                encoding="utf-8"
            )

        except PermissionError as error:

            raise RuntimeError(
                f"PWM 파일 접근 권한이 없습니다: "
                f"{path}\n"
                "프로그램을 sudo로 실행하세요."
            ) from error

        except OSError as error:

            raise RuntimeError(
                f"PWM 설정 실패: "
                f"{path} <- {value}\n"
                f"원인: {error}"
            ) from error

    # =================================
    # 안전한 파일 쓰기
    # =================================

    @staticmethod
    def _safe_write(path, value):

        try:

            path.write_text(
                str(value),
                encoding="utf-8"
            )

        except (
            OSError,
            PermissionError
        ):

            pass

    # =================================
    # Cleanup
    # =================================

    def cleanup(self):

        try:

            # ---------------------------------
            # 두 LED Duty 0%
            # ---------------------------------

            self._safe_write(
                self.white_pwm_path /
                "duty_cycle",
                0
            )

            self._safe_write(
                self.yellow_pwm_path /
                "duty_cycle",
                0
            )

            # ---------------------------------
            # PWM OFF
            # ---------------------------------

            self._safe_write(
                self.white_pwm_path /
                "enable",
                0
            )

            self._safe_write(
                self.yellow_pwm_path /
                "enable",
                0
            )

            self.enabled = False

            self.current_brightness = 0

            print(
                "[LED] Hardware PWM stopped"
            )

            # ---------------------------------
            # 내가 export한 채널만 unexport
            # ---------------------------------

            if self.white_exported_by_me:

                self._safe_write(
                    self.chip_path / "unexport",
                    self.white_channel
                )

            if self.yellow_exported_by_me:

                self._safe_write(
                    self.chip_path / "unexport",
                    self.yellow_channel
                )

        except Exception as error:

            print(
                f"[LED] Cleanup error: {error}"
            )
