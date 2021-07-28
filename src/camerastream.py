'''
    Run camera stream in thread to grab entrance frames
'''
from threading import Thread
import cv2

class CameraVideoStream:
    def __init__(self, width=640, height=480, src=0):
        
        gst_str = ('nvarguscamerasrc ! '
                    'video/x-raw(memory:NVMM), '
                    'width=(int){}, height=(int){}, '
                    'format=(string)NV12, framerate=(fraction)30/1 ! '
                    'nvvidconv flip-method=2 ! '
                    'video/x-raw, width=(int){}, height=(int){}, '
                    'format=(string)BGRx ! '
                    'videoconvert ! appsink').format(width, height, width, height)
        self.stream = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
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
            if self.grabbed:
                raw_frame = cv2.flip(raw_frame, 1)
                raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                self.frame = cv2.resize(raw_frame, (600,900))
            else:
                break

    def read(self):
        return self.frame

    def stop(self):       
        self.stopped = True
        self.stream.release()

