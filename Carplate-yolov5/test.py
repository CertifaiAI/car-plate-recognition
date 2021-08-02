'''
    The test file is to compare the execution time and fps of the program 
'''

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
from functions import tensor2List, drawBoundingBox, checkVehicleandPlatePresent, crop_image, cv2Img_base64Img, show_fps, extract_class, checkVehicleCloseEnough
from config import Config
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true', default=False, help='show results')    
parser.add_argument('--sensor', action='store_true', default=False, help='use ultrasonic sensor')
parser.add_argument('--nano', action='store_true', default=False, help='use nano')
parser.add_argument('--relay', action='store_true', default=False, help='use relay to control gate')
parser.add_argument('--led', action='store_true', default=False, help='use led')
parser.add_argument('--cudnn', action='store_true', default=False, help='use cudnn acceleration')

args = parser.parse_args()

if args.cudnn:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

# Classes
config = Config()
# door = doorControl()
# sensor = Ultrasonic()
torch.cuda.is_available()
detector = detectYolo(weight=config.WEIGHTS_PATH, device=config.DEVICE)

# Threads
# status = StatusReport(config=config, camera=camera, door=door)
camera = CameraVideoStream(nano=args.nano).start()

def loop_and_detect(camera, detector, config, show):
    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0

    # 10 inference
    inf = []
    while True:
        # get current frame
        input_frame = camera.read()

        # inference start
        inf_start = time.time()
        # Detect object with incoming video stream
        predictions, classNames = detector.inference(input_frame)

        total_inf = time.time() - inf_start

        # rescale prediction boxes
        predictions = detector.rescale_box(predictions, input_frame.shape)

        # convert tensor to lists
        predictions = tensor2List(predictions)

        # Set predictions and classNames to camera class
        camera.set_results(predictions, classNames)

        # Check vehicle and license plate detected and close enough
        if checkVehicleandPlatePresent(predictions) and checkVehicleCloseEnough(predictions):

            # set into inference list
            if len(inf) < 10: 
                inf.append(total_inf)
            else:
                print(inf)
                mean_inf = sum(inf) / len(inf)
                print("Mean of 10 inference : {:.2f} seconds".format(mean_inf))
                    # Exited 
                camera.stop()
                cv2.destroyAllWindows()
                print("Cargate Exited!")
                time.sleep(1)
                break

            # Crop plate image
            plate_image = crop_image(image=input_frame, predictions=predictions)

            # Convert cv2 image to base64 str
            plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
            plate_image_base64 = cv2Img_base64Img(plate_image)

            # Send cropped plate to server -> returned with plate number 
            try:
                data = {"image": plate_image_base64}
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
        loop_and_detect(detector=detector, camera=camera, config=config, show=args.show)
    except (KeyboardInterrupt, SystemExit):
        print('Received keyboard interrupt, quitting threads.\n')
        camera.stop()
        cv2.destroyAllWindows()
        print("Cargate Exited!")
        time.sleep(1)