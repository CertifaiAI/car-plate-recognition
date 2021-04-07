import os
from os import path
import time
import argparse
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.yolo_with_plugins import TrtYOLO
from utils.functions import carCloseEnough, carStopped, crop_plate, colorID, draw_bboxes, car_plate_present, crop_car
import model.alpr as alpr
import requests
import json

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-s', '--save', default=False, const=True, nargs='?', help='save video in .mp4')
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, save, vidwritter, prev_box, WINDOW_NAME):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    
    # FPS
    fps = 0.0
    tic = time.time()
    ALLOW = False
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        
        # detect frame return box, confidence, class
        boxes, confs, clss = trt_yolo.detect(img, conf_th=0.5)
        # check car distance
        carDistanceCloseEnough = carCloseEnough(boxes=boxes, clss=clss, distance_car_cam= 500)
        # initialize variable
        plate_number = ''
        car_color = ''
        if carDistanceCloseEnough:
            # check car stopped
            carStop = carStopped(prev_box=prev_box, boxes=boxes, clss=clss, iou_percentage=0.9)
            if carStop:
                # Both car and licence plate presence
                car_plate_here =  car_plate_present(clss=clss)
                if car_plate_here:
                    # Crop car
                    cropped_car = crop_car(img=img, boxes=boxes, clss=clss)
                    # Crop car plate
                    cropped_plate = crop_plate(img=img, boxes=boxes, clss=clss)
                    # Data
                    data = {'carShape': cropped_car.shape, 'carImg': base64.b64encode(cropped_car.tobytes()),'plateShape': cropped_plate.shape, 'plateImg': base64.b64encode(cropped_plate.tobytes())}
                    # Send request to server
                    plate_number, car_color= requests.post(server, data=json.dumps(data))
                    # Make decision based on plate number
                    if plate_number in registered_plates:
                        print("Allow access and open gate for {}".format(plate_number))
                        # Open gate 
                    else:
                        print("Access denied and not open gate")
                    # Visualize on frame with all data
                    img = draw_bboxes(img=img, boxes=boxes, confs=confs, clss=clss, lp=plate_number, carColor=car_color)
                #img = draw_bboxes(img=frame, boxes=boxes, confs=confs, clss=clss)
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)

        # data
        data = {'Plate Number': plate_number, 'Color': car_color, 'Make': '', 'Model': ''}
        print(data)

        # Save Video
        if save:
            vidwritter.write(img)
        
        # FPS calculations
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    # Initialize previous bounding boxes
    prev_box = [0,0,0,0]

    # directory to store 
    cwd = os.getcwd()
    if not path.exists(cwd+'/detections'):
        os.mkdir(cwd+'/detections')
    
    # initialize camera 
    args = parse_args()
    cam = Camera(args)

    # initialize video save
    if args.save:
        FILE_OUTPUT = './detections/test.mp4'
        out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), 20, (cam.img_width, cam.img_height))
    else:
        out = None
    
    # check camera
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')


    # Config Parser
    # config = configparser.ConfigParser()
    # config.read('settings.ini')
    # # get backend config
    # backend_hostname = config['Backend']['hostname']
    # backend_port = config['Backend']['port']
    # backend_endpoint = config['Backend']['get_plate_endpoint']
    # get_plate_address = 'http://{}:{}/{}'.format(backend_hostname, backend_port, backend_endpoint)

    # Initialize database connection to fetch carplates data
    # registered_plates = requests.get(get_plate_address)
    # Dummy 
    # registered_plates = ['WYQ8233', 'WHY1612']

    # Initialize detector
    h = w = int(416)
    carAndLP_model = 'lpandcar-yolov4-tiny-416'
    carAndLP_trt_yolo = TrtYOLO(carAndLP_model, (h, w), category_num=2) # Car and lp

    # Initialize output window
    WINDOW_NAME = 'Car Gate'
    open_window(WINDOW_NAME, 'Car Gate', cam.img_width, cam.img_height)
    # Start looping
    loop_and_detect(cam, carAndLP_trt_yolo, conf_th=0.9, save=args.save, vidwritter=out, prev_box=prev_box, WINDOW_NAME=WINDOW_NAME)
    
    # After loop release all resources
    cam.release()
    # if use Video saved
    if args.save:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
