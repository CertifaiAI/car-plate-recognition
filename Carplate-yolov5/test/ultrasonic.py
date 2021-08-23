import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)

TRIG = 7
ECHO = 12

print("Distance Measurement in progress")

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

while True:
    GPIO.output(TRIG, False)
    print("Waiting for sensor to settle")
    time.sleep(2)

    GPIO.output(TRIG, True)
    time.sleep(0.0001)
    GPIO.output(TRIG, False)
    #print(GPIO.input(ECHO))
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        #print(pulse_start)
    print(GPIO.input(ECHO))
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
    
    pulse_duration = pulse_end - pulse_start

    distance = pulse_duration * 17150
    distance = round(distance, 2)

    print("Distance: {}cm".format(distance))