import RPi.GPIO as GPIO


class LEDController:
    def __init__(self, pin=18, freq=1000, default_brightness=30):
        self.pin = pin
        self.freq = freq

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)

        self.pwm = GPIO.PWM(self.pin, self.freq)

        # 기본 밝기로 시작
        self.current_brightness = default_brightness
        self.pwm.start(default_brightness)

    def on(self):
        self.set_brightness(100)

    def off(self):
        self.set_brightness(0)

    def set_brightness(self, brightness):
        brightness = max(0, min(100, int(brightness)))

        if brightness == self.current_brightness:
            return

        self.current_brightness = brightness

        print(f"[LED] Brightness -> {brightness}%")
        self.pwm.ChangeDutyCycle(brightness)

    def cleanup(self):
        self.pwm.stop()
        GPIO.cleanup()