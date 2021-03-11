# Ref: https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# How do it works?
# FPS improved by create new thread that pool new frames while main thread process frame
# Increased fps over 300%

# from fps import FPS
from utils.videoStream import WebcamVideoStream
#from utils.detection import YOLODetection
import imutils
import cv2
import time
from utils.display import show_fps
import argparse
import pycuda.autoinit
from utils.yolo_with_plugins import TrtYOLO

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--width', help='width of camera captured', default=1280)
parser.add_argument('--height', help='height of camera captured', default=720)
args = parser.parse_args()

# Initialize threads and constant
h = w = int(416)
carAndLP_model = 'lpandcar-yolov4-tiny-416'
carAndLP_trt_yolo = TrtYOLO(carAndLP_model, (h, w), category_num=2)
vs = WebcamVideoStream(src=0, width=args.width, height=args.height).start()
#yolo_detect = YOLODetection(videoStream=vs, model=carAndLP_trt_yolo).start()
in_display_fps = 0.0
tic = time.time()

while(True):
    frame = vs.read()
    boxes, confs, clss = carAndLP_trt_yolo.detect(frame, conf_th=0.5)
    print(boxes)
    vs.set_yolo_result(boxes, confs, clss)
    
    frame = vs.output_yolo()
    # Draw in display fps    
    img = show_fps(frame, in_display_fps)
    #Display
    frame = imutils.resize(frame, width=700)
    cv2.imshow("Frame", img)
    
    # In display fps
    toc = time.time()
    curr_fps = 1.0 / (toc - tic)
    in_display_fps = curr_fps if in_display_fps == 0.0 else (in_display_fps*0.95 + curr_fps*0.05)
    tic = toc

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
#yolo_detect.stop()
