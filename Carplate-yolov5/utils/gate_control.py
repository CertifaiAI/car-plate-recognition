import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)


class GateControl:
	def __init__(self):
		self.output_pin = 35  # gpio76
		self.delay = 2
		# gate off
		self.gate_status = 0
		GPIO.setup(self.output_pin, GPIO.OUT, initial=GPIO.LOW)

	def relay_on(self):
		print('Relay ON')
		GPIO.output(self.output_pin, GPIO.HIGH)  # Turn relay on
		self.gate_status = 1 # gate on
		time.sleep(self.delay)
		print('Relay OFF')
		GPIO.output(self.output_pin, GPIO.LOW)  # Turn relay off
		self.gate_status = 0 # gate off
		time.sleep(self.delay)
		# GPIO.cleanup()
