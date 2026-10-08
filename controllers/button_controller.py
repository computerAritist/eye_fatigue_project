# controllers/button_controller.py

import time
import lgpio


class ButtonController:

    # -----------------------------
    # GPIO Pin
    # -----------------------------

    MODE_BUTTON = 17
    BRIGHTNESS_BUTTON = 22
    CCT_BUTTON = 27

    def __init__(self):

        # Raspberry Pi 5 GPIO chip
        self.h = lgpio.gpiochip_open(0)

        # 버튼 GPIO 설정
        buttons = [
            self.MODE_BUTTON,
            self.BRIGHTNESS_BUTTON,
            self.CCT_BUTTON
        ]

        for pin in buttons:
            lgpio.gpio_claim_input(
                self.h,
                pin,
                lgpio.SET_PULL_UP
            )

        # 이전 버튼 상태
        self.last_state = {
            self.MODE_BUTTON: 1,
            self.BRIGHTNESS_BUTTON: 1,
            self.CCT_BUTTON: 1
        }

        # 마지막 입력 시간
        self.last_time = {
            self.MODE_BUTTON: 0,
            self.BRIGHTNESS_BUTTON: 0,
            self.CCT_BUTTON: 0
        }

        # Debounce
        self.debounce_time = 0.15

    # -----------------------------
    # 버튼 하나의 눌림 확인
    # -----------------------------

    def _is_pressed(self, pin):

        current_state = lgpio.gpio_read(
            self.h,
            pin
        )

        current_time = time.monotonic()

        # HIGH -> LOW
        # 버튼이 눌린 순간
        if (
            self.last_state[pin] == 1
            and current_state == 0
            and (
                current_time - self.last_time[pin]
                > self.debounce_time
            )
        ):

            self.last_time[pin] = current_time
            self.last_state[pin] = current_state

            return True

        self.last_state[pin] = current_state

        return False

    # -----------------------------
    # 버튼 이벤트 확인
    # -----------------------------

    def get_event(self):

        if self._is_pressed(self.MODE_BUTTON):
            return "MODE"

        if self._is_pressed(self.BRIGHTNESS_BUTTON):
            return "BRIGHTNESS"

        if self._is_pressed(self.CCT_BUTTON):
            return "CCT"

        return None

    # -----------------------------
    # Cleanup
    # -----------------------------

    def cleanup(self):

        lgpio.gpiochip_close(self.h)
