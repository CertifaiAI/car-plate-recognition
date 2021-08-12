import serial
import time 

ser= serial.Serial('/dev/ttyACM0', 9600)
output = 'Hello'
time.sleep(2)
ser.write(output.encode())
