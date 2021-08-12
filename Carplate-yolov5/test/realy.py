import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
output_pin = 35  # gpio76
delay = 2
GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)
print('Relay ON')
GPIO.output(output_pin, GPIO.HIGH)  # Turn relay on
time.sleep(delay)
print('Relay OFF')
GPIO.output(output_pin, GPIO.LOW)  # Turn relay off
time.sleep(delay)
