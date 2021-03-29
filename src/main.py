''' To Do: 
1. Run video stream with threading
2. Run AI in order
3. Connect backend 

Flow:
1. Get frame from camera
2. Detect car
3. Detect plate
4. Recignize plate
5. Check with db
6. If-else plate is registered 
7. 

'''
# Imports
# camera stream
from utils.videoStream import WebcamVideoStream
import cv2 
# fps counter
import time
from utils.display import show_fps
# parsing arguement
import argparse
# parse settings
import configparser
# backend 
import requests
from datetime import datetime
# detection


# Parser
# arguement parser
parser = argparse.ArgumentParser()
parser.add_argument('--width', help='width of camera captured', default=1280)
parser.add_argument('--height', help='height of camera captured', default=720)
args = parser.parse_args()
# config parser
config = configparser.ConfigParser()
server = config['Server']['url']
port = config['Server']['port']
url = 'http://{}:{}/'.format(server, port)

# Constant
# video stream
vs = WebcamVideoStream(src=0, width=args.width, height=args.height).start()
# fps counter
in_display_fps = 0.0
tic = time.time()
# detection

# endpoints
get_plate_endpoint = url + 'carplates'
post_data_endpoint = url + 'data'

# Main 
while(True):
    # connect to backend to get plates
    try:
        # get plate numbers from db : list
        registed_lp_plates = requests.get(ip+"get_plate_endpoint")
        registed_lp_plates = registed_lp_plates.json()
    except:
        print('Cannot connect to backend server')
    # Read frame from vs thread
    frame = vs.read()
    # Draw 'in display' fps counter    
    img = show_fps(frame, in_display_fps)
    # Pass to detector : return car plate num


    # check plate is registered?
    if final_lp in registed_lp_plates:
        # Allow car enter (light up LED)
        # ser = serial.Serial('/dev/ttyACM0', 9600)
        # Post to API
        open_time = datetime.now()
        data = {
            "carplate_no": final_lp,
            "time": str(open_time)
        }
        # requests.post(ip+"enter", data=json.dumps(data))
        print("Please Enter")
    else:
        print("Permision Denied!")

    # show frame 
    frame = imutils.resize(frame, width=700)
    cv2.imshow("Frame", img)
    # in display fps
    toc = time.time()
    curr_fps = 1.0 / (toc - tic)
    in_display_fps = curr_fps if in_display_fps == 0.0 else (in_display_fps*0.95 + curr_fps*0.05)
    tic = toc
    # break when interupt
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# get rid of cv2 windows
cv2.destroyAllWindows()
# stop thread
vs.stop()