from threading import Thread
import time
import cv2
from utils.yolo_with_plugins import TrtYOLO

class YOLODetection:
    def __init__(self, videoStream):
        self.videoStream = videoStream
        self.thread = None
        self.stopped = False
        h = w = int(416)
        carAndLP_model = 'lpandcar-yolov4-tiny-416'
        self.carAndLP_trt_yolo = TrtYOLO(carAndLP_model, (h, w), category_num=2)
    
    def start(self):
        self.stopped = False
        self.thread = Thread(target=self.update)
        self.thread.start()
        return self
    
    def update(self):
        while True:
            #time.sleep(0.1)
            if self.stopped:
                return
            frame = self.videoStream.read()
            # Detection here
            try:
                boxes, confs, clss = self.carAndLP_trt_yolo.detect(frame, conf_th=0.5)
                self.videoStream.set_yolo_result(boxes, confs, clss)
            except:
                self.videoStream.set_yolo_result(None,None, None)
            #time.sleep(0.1)
            self.videoStream.set_yolo_result(None,None, None)
    
    def stop(self):
        self.stopped = True