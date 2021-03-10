from threading import Thread
import cv2
import subprocess
from visualization import BBoxVisualization

def open_cam_onboard(width, height):
    """Open the Jetson onboard camera."""
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, you might need to add
        # 'flip-method=2' into gst_str below.
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

class WebcamVideoStream:
  def __init__(self, src=0, width ,height):
    self.stream = open_cam_onboard(width=width, height=height)
    (self.grabbed, self.frame) = self.stream.read()
    self.stopped = False
    self.boxes = None
    self.confs = None
    self.clss = None
    car_custom_dict = {0 : 'Car', 1: 'Licence Plate'}
    self.vis = BBoxVisualization(car_custom_dict)
  
  def start(self):
    Thread(target=self.update, args=()).start()
    return self

  def update(self):
    while True:
      if self.stopped:
        return
      (self.grabbed, self.frame) = self.stream.read()

  def read(self):
    return self.frame

  def stop(self):
    self.stopped = True
  
  def set_yolo_result(self, boxes, confs, clss):
    self.boxes = boxes
    self.confs = confs
    self.clss = clss

  def get_yolo_result(self):
    return (self.boxes, self.confs, self.clss)

  def output_yolo(self):
    output = self.frame
    if self.boxes and self.confs and self.clss is not None:
      # Crop car plate
      cropped = self.vis.crop_plate(output, self.boxes, self.confs, self.clss)
      # Visualize image
      output = self.vis.draw_bboxes(output, boxes, confs, clss, lp= '')
    return output

