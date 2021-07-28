'''
    Cargate program Implementation on Jetson Nano
'''
# from ultrasonicSensor import Ultrasonic
from camerastream import CameraVideoStream
from detectYolov5 import detectYolo
# from statusReport import StatusReport
import argparse
import cv2
# import json
# import requests
from functions import tensor2List, drawBoundingBox, checkVehicleandPlatePresent, crop_image, cv2Img_base64Img, show_fps, extract_class
from config import Config
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true', default=False, help='show results')    
parser.add_argument('--sensor', action='store_true', default=False, help='use ultrasonic sensor')
parser.add_argument('--nano', action='store_true', default=False, help='use nano')    
args = parser.parse_args()

# Classes
config = Config()
# sensor = Ultrasonic()
torch.cuda.is_available()
detector = detectYolo(weight=config.WEIGHTS_PATH, device=config.DEVICE)

# Threads 
# status = StatusReport(config=config)
camera = CameraVideoStream(nano=args.nano)
# start threads
camera.start()

def loop_and_detect(camera, detector, config, show):
    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
    while True:
        # get current frame
        input_frame = camera.read()

        # Detect object with incoming video stream
        predictions, classNames = detector.inference(input_frame)
    
        # rescale prediction boxes
        predictions = detector.rescale_box(predictions, input_frame.shape)

        # convert tensor to lists
        predictions = tensor2List(predictions)

        # process predictions
        allClass = extract_class(predictions)

        # If vehicle and license plate detected. crop license plate
        if checkVehicleandPlatePresent(allClass):
            plate_image = crop_image(image=input_frame, predictions=predictions)

            # Convert cv2 image to base64 str
            plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
            plate_image_base64 = cv2Img_base64Img(plate_image)
            print(plate_image_base64)

            # Send cropped plate to server -> returned with plate number 
            try:
                data = {"image": plate_image_base64}
                print(data)
                # response = requests.post(config.SERVER_URL, data=json.dumps(data))
                # result = response.text
            except:
                print("Failed to send plate to server")
            
            # # Need authorized + plate number
            # if result is not None:
            #     # Process result from server -> show on LED screen 

            #     # Send data to LED panel
            #     pass
        # show result
        if show:
            # draw bounding box on frame
            drawBoundingBox(input_frame, predictions, classNames)
            
            # FPS calculation
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            img = show_fps(input_frame, fps)

            cv2.imshow("Frame", input_frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Exited 
                camera.stop()
                cv2.destroyAllWindows()
                print("Cargate Exited!")
                time.sleep(1)
                break

if __name__ == '__main__':
    print("Running Cargate now...")
    try:
        loop_and_detect(detector=detector, camera=camera, config=config, show=args.show)
    except (KeyboardInterrupt, SystemExit):
        print('Received keyboard interrupt, quitting threads.\n')
        camera.stop()
        cv2.destroyAllWindows()
        print("Cargate Exited!")
        time.sleep(1)