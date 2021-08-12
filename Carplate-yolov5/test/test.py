import smbus
import time
bus = smbus.SMBus(1)
address = 8 # I2C address for Ultrasonic bus

while True:
    print(bus.read_byte_data(address, 0))
    time.sleep(1)
