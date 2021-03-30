from threading import Thread
import pycuda.autoinit
import time
import cv2
from utils.yolo_with_plugins import TrtYOLO
# import yolo_with_plugins.TrtYOLO

class YOLODetection:
    def __init__(self, videoStream, modelName, classNames):
        self.videoStream = videoStream
        # self.modelPath = modelPath
        self.thread = None
        self.stopped = False
        self.classNames = classNames
        self.trt_yolo = TrtYOLO(modelName, (416, 416), 1, True)
        
    def start(self):
        self.stopped = False
        self.thread = Thread(target=self.update)
        self.thread.start()
        return self
    
    def update(self):
        print('started')
        while True:
            time.sleep(1)
            if self.stopped:
                return
            frame = self.videoStream.read()
            # Detection here
            try:
                boxes, confs, clss = self.trt_yolo.detect(frame, conf_th=0.5)
                print(boxes)
                # self.videoStream.set_yolo_result(boxes, confs, clss)
            except:
                # self.videoStream.set_yolo_result(None,None, None)
                print('Cannot detect')
            #time.sleep(0.1)
            # self.videoStream.set_yolo_result(None,None, None)
    
    def stop(self):
        self.stopped = True
