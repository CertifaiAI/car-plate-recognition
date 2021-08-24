'''
    Config class for cargate configurations
'''

class Config: 
    def __init__(self):
        # Server
        self.SERVER_URL = 'http://139.162.33.89:8080/api/v1/r69C44coGlPTANdNBnqf/rpc'

        # Model
        self.WEIGHTS_PATH = 'yolov5/weights/detection.pt'
        self.DEVICE = 'cuda'

        # Sensor
        self.SENSOR_DIST = 300

        # Status report 
        # self.STATUS_URL = ''
        
        # TOKEN
        self.STATUS_TOKEN = ''
        self.STATUS_HOST = ''
        self.STATUS_PORT = ''
        self.STATUS_END_POINT = ''

    
    def writeSecret(self):
        pass

    def readSecret(self):
        pass
