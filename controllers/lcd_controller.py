# controllers/lcd_controller.py

from RPLCD.i2c import CharLCD


class LCDController:

    # I2C 설정
    I2C_PORT = 8
    I2C_ADDRESS = 0x27

    def __init__(self):

        self.lcd = CharLCD(
            i2c_expander="PCF8574",
            address=self.I2C_ADDRESS,
            port=self.I2C_PORT,
            cols=16,
            rows=2,
            charmap="A00",
            auto_linebreaks=False
        )

        self.previous_lines = None

        self.lcd.clear()

        print("[LCD] I2C Initialized")

    # 밝기 → 레벨 변환
    @staticmethod
    def _brightness_to_level(brightness):

        brightness = int(brightness)

        if brightness <= 20:
            return 1
        elif brightness <= 40:
            return 2
        elif brightness <= 60:
            return 3
        elif brightness <= 80:
            return 4
        else:
            return 5

    # LCD 한 줄 출력
    def _write_line(self, row, text):

        text = str(text)[:16].ljust(16)

        self.lcd.cursor_pos = (row, 0)
        self.lcd.write_string(text)

    # LCD 상태 업데이트
    def update(self, mode, status, brightness, cct_mode):

        mode = str(mode).upper()
        cct_mode = int(cct_mode)

        level = self._brightness_to_level(brightness)

        if mode == "AUTO":

            line1 = "AUTO"

            if status == "Measuring...":
                line2 = "Measuring..."
            else:
                line2 = f"{status} L{level} C{cct_mode}"

        else:

            line1 = "MANUAL"
            line2 = f"L{level} C{cct_mode}"

        # 화면 내용이 변경될 때만 갱신
        current_lines = (line1, line2)

        if current_lines == self.previous_lines:
            return

        self._write_line(0, line1)
        self._write_line(1, line2)

        self.previous_lines = current_lines

    # 종료 처리
    def cleanup(self):

        try:
            self.lcd.clear()
            self.lcd.close()
        except Exception as e:
            print(f"[LCD] Cleanup error: {e}")

        print("[LCD] Cleanup complete")
