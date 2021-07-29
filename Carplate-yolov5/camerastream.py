'''
    Run camera stream in thread to grab entrance frames
'''
from threading import Thread
import cv2
from torch._C import set_flush_denormal
from functions import drawBoundingBox

class CameraVideoStream:
    def __init__(self, width=640, height=480, src=0, nano=True):
        
        gst_str = ('nvarguscamerasrc ! '
                    'video/x-raw(memory:NVMM), '
                    'width=(int){}, height=(int){}, '
                    'format=(string)NV12, framerate=(fraction)30/1 ! '
                    'nvvidconv flip-method=2 ! '
                    'video/x-raw, width=(int){}, height=(int){}, '
                    'format=(string)BGRx ! '
                    'videoconvert ! appsink').format(width, height, width, height)

        if nano:
            self.stream = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        else:
            self.stream = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.stream.read()
        self.result = None
        self.predictions = None
        self.stopped = False
        self.classes = None
        self.thread = None

    def start(self):
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        print("Camera thread starting")
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, raw_frame) = self.stream.read()
            if self.predictions is not None:
                self.draw_box()
            if self.grabbed:
                self.frame = cv2.resize(raw_frame, (600,500))
            else:
                break   

    def read(self):
        return self.frame

    def stop(self):       
        self.stopped = True
        self.stream.release()

    def draw_box(self):
        image = drawBoundingBox(self.frame, self.predictions, self.classes)
        self.result = image

    def set_results(self, predictions, classes):
        self.predictions = predictions
        self.classes = classes

if __name__ == "__main__":
    camera = CameraVideoStream(nano=False)
    camera.start()
    while (True):
        Webrame = camera.read()
        cv2.imshow('Result', Webrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.stop()
    cv2.destroyAllWindows()
