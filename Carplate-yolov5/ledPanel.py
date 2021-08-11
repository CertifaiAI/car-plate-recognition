'''
    Send data to bluetooth device to display data on P10 LED Panel
'''

import serial

class LedPanel():
    def __init__(self):
        # Set USB port for arduino
        self.ser= serial.Serial('/dev/ttyACM0', 9600)
        self.welcome_message = 'Welcome to Skymind'

    def send_data(self, data):
        # Send to LED
        self.ser.write(data.encode())
        # Write back to ori
        self.ser.write(self.welcome_message.encode())