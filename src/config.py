'''
    Config class for cargate configurations
'''

class Config: 
    def __init__(self):
        self.SERVER_URL = ''
        self.STATUS_SERVER = ''
        self.WEIGHTS_PATH = 'yolov5/weights/detection.pt'
        self.DEVICE = 'cuda'
        self.SENSOR_DIST = 100