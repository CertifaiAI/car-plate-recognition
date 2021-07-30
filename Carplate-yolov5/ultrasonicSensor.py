import smbus
import time

class Ultrasonic:
    def __init__(self):
        self.bus = smbus.SMBus(1)
        self.address = 8 # I2C address for Ultrasonic bus

    def get_distance(self):
        return self.bus.read_byte_data(self.address, 0)