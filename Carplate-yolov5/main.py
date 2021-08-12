'''
    Cargate program Implementation on Jetson Nano
'''
from utils.camerastream import CameraVideoStream
from utils.detectYolov5 import detectYolo
from utils.statusReport import StatusReport
import argparse
import cv2
import json
import requests
from utils.functions import tensor2List, checkVehicleandPlatePresent, crop_image, cv2Img_base64Img, show_fps, checkVehicleCloseEnough
from utils.config import Config
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true', default=False, help='show results')    
parser.add_argument('--sensor', action='store_true', default=False, help='use ultrasonic sensor')
parser.add_argument('--nano', action='store_true', default=False, help='use nano')
parser.add_argument('--relay', action='store_true', default=False, help='use relay to control gate')
parser.add_argument('--led', action='store_true', default=False, help='use led')
parser.add_argument('--server', action='store_true', default=False, help='use backend server for ppocr inference')

args = parser.parse_args()

# Classes
config = Config()

# Threads
camera = CameraVideoStream(nano=args.nano).start()

if args.nano:        
    from utils.ledPanel import LedPanel
    from utils.gate_control import GateControl
    from utils.ultrasonicSensor import Ultrasonic
    gate = GateControl()
    sensor = Ultrasonic()
    ledPanel = LedPanel()
    status = StatusReport(config=config, camera=camera, door=gate)

torch.cuda.is_available()
detector = detectYolo(weight=config.WEIGHTS_PATH, device=config.DEVICE)



def loop_and_detect(camera, detector, config):
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

        # Set predictions and classNames to camera class
        camera.set_results(predictions, classNames)

        # Check vehicle and license plate detected and close enough
        if checkVehicleandPlatePresent(predictions) and checkVehicleCloseEnough(predictions):

            # Crop plate image
            plate_image = crop_image(image=input_frame, predictions=predictions)

            # Convert cv2 image to base64 str
            plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
            plate_image_base64 = cv2Img_base64Img(plate_image)

            # Send cropped plate to server -> returned with plate number
            if args.server:
                try:
                    data = {"image": plate_image_base64}
                    response = requests.post(config.SERVER_URL, data=json.dumps(data))
                    result = response.text
                    # TODO find out how result is returned, extract authentication, plate number
                except:
                    print("Failed to send plate to server")
                
                # TODO: add authentication result
                # Need authorized + plate number
                if result is not None and args.led:
                    # Process result from server -> show on LED screen 
                    # TODO replace result with plate data
                    ledPanel.send_data(result)
                    if args.relay:
                        gate.relay_on()
        # show result
        if args.show:
            if camera.result is not None:
                result = camera.result

                # FPS calculation
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                fps = str(int(fps))
                img = show_fps(result, fps)

                cv2.imshow("Result", result)

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
        # Ultrasonic sensor here
        if args.sensor and args.nano:
            if sensor.get_distance() < 200: # less 200 cm has object
                loop_and_detect(detector=detector, camera=camera, config=config)
        else:
            loop_and_detect(detector=detector, camera=camera, config=config)
    except (KeyboardInterrupt, SystemExit):
        print('Received keyboard interrupt, quitting threads.\n')
        camera.stop()
        cv2.destroyAllWindows()
        print("Cargate Exited!")
        time.sleep(1)