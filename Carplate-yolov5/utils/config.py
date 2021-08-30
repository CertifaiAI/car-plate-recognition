'''
    Config class for cargate configurations
'''

class Config: 
    def __init__(self):
        # URL for kraftboard server
        self.SERVER_URL = 'http://139.162.33.89:8080/api/v1/cCTjoonMOneHUkz4iDdI/rpc'

        # Model path 
        self.WEIGHTS_PATH = 'yolov5/weights/detection.pt'
        # Model settings 
        self.DEVICE = 'cpu'

        # Sensor distance (object to device)
        self.SENSOR_DIST = 300

        # Status report 
        # self.STATUS_URL = ''
        
        # # TOKEN
        # self.STATUS_TOKEN = ''
        # self.STATUS_HOST = ''
        # self.STATUS_PORT = ''
        # self.STATUS_END_POINT = ''
