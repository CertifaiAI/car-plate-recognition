import os
from os import path
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

from utils.yolo_with_plugins import TrtYOLO

# OCR
import model.alpr as alpr
from statistics import mode

lpr = alpr.AutoLPR(decoder='bestPath', normalise=True)
lpr.load(crnn_path='model/weights/best-fyp-improved.pth')

# Constant 
WINDOW_NAME = 'TrtYOLODemo'
lp_database = ['PEN1234', 'SCE5678']
FILE_OUTPUT = './detections/test.mp4'
# loop 5 times for mode result
#list_of_plates = []
#colected_lp = 5

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-s', '--save', default=False, const=True, nargs='?', help='save video')
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis, save, vidwritter):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    
    # FPS
    #fps = 0.0
    #tic = time.time()
    ALLOW = False
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
       
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        cropped = vis.crop_plate(img, boxes, confs, clss)
        lp_plate = ''
        if cropped is not None:
           lp_plate = lpr.predict(cropped)
           # Take directly
           if lp_plate in lp_database:
              # Open Door
              print('Door Opened for {}'.format(lp_plate))
              # Set allow to change box colour to green
              ALLOW = True
           else:
              print('Access Denied!')        


        # Loop 5 times then get mode (final) 
        #if len(list_of_plates) < colected_lp:
           #list_of_plates.append(lp_plate)
           #print(list_of_plates)
        #else:
           #try:
                #final = mode(list_of_plates)
           #except:
                #print('No mode found')
           #list_of_plates.clear()
           #print("Final Car Plate Result: {}".format(final))
           # Compare with db
           #if final in lp_database:
              # Open Door
              #print('Door Opened for {}'.format(final))
              #ALLOW = True
           #else:
              #print('Access Denied!')
           #final = ''
           
   
        img = vis.draw_bboxes(img, boxes, confs, clss, lp= lp_plate, allow=ALLOW)
        ALLOW = False
        #img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        if save:
            vidwritter.write(img)
        
        # FPS calculations
        #toc = time.time()
        #curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        #fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        #tic = toc
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    # Directory to store results
    cwd = os.getcwd()
    if not path.exists(cwd+'/detections'):
        os.mkdir(cwd+'/detections')
   
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    #if not os.path.isfile('yolo/%s.trt' % args.model):
        #raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if args.save:
        out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), 20, (cam.img_width, cam.img_height))
    else:
        out = None
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
    # Replace with custom dict
    car_custom_dict = {0 : 'Car', 1: 'Licence Plate'}
    h = w = int(416)
    carAndLP_model = 'lpandcar-yolov4-tiny-416'
    carAndLP_trt_yolo = TrtYOLO(carAndLP_model, (h, w), 2, args.letter_box)

    open_window(
        WINDOW_NAME, 'Car and License Plate Detector',
        cam.img_width, cam.img_height)
    vis = BBoxVisualization(car_custom_dict, vid=args.save)
    loop_and_detect(cam, carAndLP_trt_yolo, conf_th=0.5, vis=vis, save=args.save, vidwritter=out)
    
    cam.release()
    if args.save:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
