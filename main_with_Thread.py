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
import model.alpr as alpr
# import requests
# import configparser
from utils.functions import carCloseEnough, carStopped, crop_plate, colorID, draw_bboxes, car_plate_present, crop_car
# Arguement Parser
parser = argparse.ArgumentParser()
parser.add_argument('--width', help='width of camera captured', default=1280)
parser.add_argument('--height', help='height of camera captured', default=720)
args = parser.parse_args()

# Config Parser
# config = configparser.ConfigParser()
# config.read('settings.ini')
# # get backend config
# backend_hostname = config['Backend']['hostname']
# backend_port = config['Backend']['port']
# backend_endpoint = config['Backend']['get_plate_endpoint']
# get_plate_address = 'http://{}:{}/{}'.format(backend_hostname, backend_port, backend_endpoint)

# Initialize detector
h = w = int(416)
carAndLP_model = 'lpandcar-yolov4-tiny-416'
carAndLP_trt_yolo = TrtYOLO(carAndLP_model, (h, w), category_num=2) # Car and lp

# Initialze recognizer
lpr = alpr.AutoLPR(decoder='bestPath', normalise=True)
lpr.load(crnn_path='model/weights/best-fyp-improved.pth')

# Initialize video stream
vs = WebcamVideoStream(src=0, width=args.width, height=args.height).start()

# Initialize fps 
in_display_fps = 0.0
tic = time.time()

# Initialize previous bounding boxes
prev_box = [0,0,0,0]

# Initialize database connection to fetch carplates data
# registered_plates = requests.get(get_plate_address)
# Dummy 
# registered_plates = ['WYQ8233', 'WHY1612']

while(True):
    # read frame from videostream
    frame = vs.read()
    # detect frame return box, confidence, class
    boxes, confs, clss = carAndLP_trt_yolo.detect(frame, conf_th=0.9)
    # check car distance
    carDistanceCloseEnough = carCloseEnough(boxes=boxes, clss=clss, distance_car_cam= 800)
    if carDistanceCloseEnough:
        # check car stopped
        carStop = carStopped(prev_box=prev_box, boxes=boxes, clss=clss, iou_percentage=0.98)
        if carStop:
            # Both car and licence plate presence
            car_plate_here =  car_plate_present(clss=clss)
            if car_plate_here:
                plate_number = ''
                car_color = ''
                # Crop car
                # cropped_car = crop_car(img=frame, boxes=boxes, clss=clss)
                # Crop car plate
                cropped_plate = crop_plate(img=frame, boxes=boxes, clss=clss)
                # Start recognize plate
                plate_number = lpr.predict(cropped_plate)
                # Do color identification
                # car_color = colorID(img=cropped_car, NUM_CLUSTERS=2)
                # print(car_color)
            #     # Maybe car make and model

            #     # Make decision based on plate number
            #     if plate_number in registered_plates:
            #         print("Allow access and open gate for {}".format(plate_number))
            #         # Open gate 
            #     else:
            #         print("Access denied and not open gate")
                # Visualize on frame with all data
                frame = draw_bboxes(img=frame, boxes=boxes, confs=confs, clss=clss, lp=plate_number, carColor=car_color)
            # frame = draw_bboxes(img=frame, boxes=boxes, confs=confs, clss=clss)
    # set result to vs to draw boxes
    # vs.set_yolo_result(boxes, confs, clss)

    # Get drawed box frame from vs 
    # frame = vs.output_yolo()

    # Draw in display fps    
    img = show_fps(frame, in_display_fps)

    #Display frame
    frame = imutils.resize(frame, width=1000)
    cv2.imshow("Frame", frame)
    
    # In display fps
    toc = time.time()
    curr_fps = 1.0 / (toc - tic)
    in_display_fps = curr_fps if in_display_fps == 0.0 else (in_display_fps*0.95 + curr_fps*0.05)
    tic = toc

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()

