'''
    Cargate program Implementation on Jetson Nano
'''
from ultrasonicSensor import Ultrasonic
from camerastream import CameraVideoStream
from detectYolov5 import detectYolo
from statusReport import StatusReport
import argparse
import cv2
import json
import requests
from functions import tensor2List, drawBoundingBox, process_predictions, checkVehicleandPlatePresent, crop_image, cv2Img_base64Img
from config import Config
import time

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true', default=False, help='show results')    
parser.add_argument('--sensor', action='store_true', default=False, help='show results')    
args = parser.parse_args()

# Classes
config = Config()
sensor = Ultrasonic()
detector = detectYolo(weight=config.WEIGHTS_PATH, device=config.device)

# Threads 
status = StatusReport(config=config)
camera = CameraVideoStream()
camera.start()

def loop_and_detect(camera, detector, config, show):
    # get current frame
    curFrame = camera.read()
    # Detect object with incoming video stream
    predictions, classNames = detector.inference(curFrame)
    
    # rescale prediction boxes
    predictions = detector.rescale_box(predictions, curFrame.shape)

    # convert tensor to lists
    predictions = tensor2List(predictions)

    # process predictions
    data, allClass = process_predictions(predictions, classNames)

    # If vehicle and license plate detected. crop license plate
    if checkVehicleandPlatePresent(allClass):
        plate_image = crop_image(image=curFrame, data=data)

        # Convert cv2 image to base64 str
        plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
        plate_image = cv2Img_base64Img(plate_image)

        # Send cropped plate to server -> returned with plate number 
        try:
            data = {"image": plate_image}
            response = requests.post(config.SERVER_URL, data=json.dumps(data))
            result = response.text
        except:
            print("Failed to send plate to server")
        
        # Need authorized + plate number
        if result is not None:
            # Process result from server -> show on LED screen 

            # Send data to LED panel
            pass
        
    # show result
    if show:
        # draw bounding box on frame
        drawBoundingBox(curFrame, predictions, classNames)
        cv2.imshow("Result", curFrame)

if __name__ == '__main__':
    print("Running Cargate now...")
    try:
        while True:
            if args.sensor:
                if (sensor.get_distance() < config.SENSOR_DIST):
                    loop_and_detect(detector=detector, camera=camera, config=config, show=args.show)
            else:
                loop_and_detect(detector=detector, camera=camera, config=config, show=args.show)
    except (KeyboardInterrupt, SystemExit):
        print('Received keyboard interrupt, quitting threads.\n')
        camera.stop()
        print("Cargate Exited!")
        time.sleep(1)