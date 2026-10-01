from RPLCD.i2c import CharLCD

class LCDController:

    def __init__(self):

        self.lcd = CharLCD(
            i2c_expander='PCF8574',
            address=0x27,
            port=1,
            cols=16,
            rows=2
        )

        self.lcd.clear()

    def display(self, fatigue_status, brightness_level):

        self.lcd.clear()

        self.lcd.write_string(f"Fatigue:{fatigue_status}")

        self.lcd.cursor_pos = (1, 0)
        self.lcd.write_string(f"Brightness:{brightness_level}")

    def clear(self):
        self.lcd.clear()