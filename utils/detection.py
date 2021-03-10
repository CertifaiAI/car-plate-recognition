from threading import Thread
import time
import cv2

class YOLODetection:
    def __init__(self, videoStream, model):
        self.videoStream = videoStream
        self.modeltrt = model
        self.thread = None
        self.stopped = False
        
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
                boxes, confs, clss = self.modeltrt.detect(frame, conf_th=0.5)
                self.videoStream.set_yolo_result(boxes, confs, clss)
            except:
                self.videoStream.set_yolo_result(None,None, None)
            #time.sleep(0.1)
            self.videoStream.set_yolo_result(None,None, None)
    
    def stop(self):
        self.stopped = True
