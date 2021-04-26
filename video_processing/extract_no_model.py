from absl import app
from absl import flags
import cv2
import os
from os import path
import time

FLAGS = flags.FLAGS
flags.DEFINE_string('video', None, 'path to input video')
flags.DEFINE_integer('interval', 10, 'interval (in seconds) between captured frames')

def main(argv):
    vid_path = FLAGS.video
    interval = FLAGS.interval
    counter = 1
    saved_time = 0
    frame_id = 0

    cwd = os.getcwd()
    if not path.exists(cwd+'/images'):
        os.mkdir(cwd+'/images')
    
    print("Processing Video From: ", vid_path)
    vid = cv2.VideoCapture(vid_path)
    while True:
        return_value, frame = vid.read()
        if not return_value:
            print("Video processing complete")
            break
        # Current time
        curr_time = time.time()
        # Cal difference in time
        diff = curr_time - saved_time
        if diff > interval:
            # Save image to folder
            cv2.imwrite('images/image{}.jpg'.format(counter), frame)
            print("Image{}.jpg is saved on image folder".format(counter))
            counter += 1
            # Saved time
            saved_time = time.time()
        
        # Show video
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass