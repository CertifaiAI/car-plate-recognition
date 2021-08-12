'''
    Send data to bluetooth device to display data on P10 LED Panel
'''

import serial
import time

class LedPanel():
    def __init__(self):
        # Set USB port for arduino
        try:
            self.ser= serial.Serial('/dev/ttyACM0', 9600)
            self.welcome_message = 'Welcome'
        except:
            print("Cannot connect to arduino")
        # try:
        #     self.ser= serial.Serial('/dev/ttyUSB0', 9600)
        #     self.welcome_message = 'Welcome'
        # except:
        #     print("Cannot connect to arduino")

    def send_data(self, data):
        # Send to LED
        output = 'Welcome ' + data
        # Allow connection establish
        time.sleep(2)
        self.ser.write(output.encode())
        # Write back to ori
        time.sleep(2)
        self.ser.write(self.welcome_message.encode())