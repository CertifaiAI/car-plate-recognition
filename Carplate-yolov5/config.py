'''
    Config class for cargate configurations
'''

class Config: 
    def __init__(self):
        # Server
        self.SERVER_URL = ''

        # Model
        self.WEIGHTS_PATH = 'yolov5/weights/detection.pt'
        self.DEVICE = 'gpu'

        # Sensor
        self.SENSOR_DIST = 100

        # Status report 
        self.STATUS_URL = ''
        
        # TOKEN
        self.STATUS_TOKEN = ''
        self.STATUS_HOST = ''
        self.STATUS_PORT = ''
        self.STATUS_END_POINT = ''

    
    def writeSecret(self):
        pass

    def readSecret(self):
        pass
