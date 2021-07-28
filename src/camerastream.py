'''
    Run camera stream in thread to grab entrance frames
'''
from threading import Thread
import cv2

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
        self.stream = cv2.VideoCapture(0)
        if nano:
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
                self.frame = cv2.resize(raw_frame, (600,500))
            else:
                break

    def read(self):
        return self.frame

    def stop(self):       
        self.stopped = True
        self.stream.release()

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
