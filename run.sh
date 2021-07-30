#!/bin/bash
xhost +local:
#sudo docker run --privileged -it --rm --runtime nvidia --network host --device /dev/video0:/dev/video0:mrw -e DISPLAY=$DISPLAY -v /tmp/argus_socket:/tmp/argus_socket -v /tmp/.X11-unix/:/tmp/.X11-unix -v /home/skymind/car-plate-recognition/Carplate-yolov5/weights:/car-plate-recognition/Carplate-yolov5/weights cargate python3 send_video.py --source 0

#sudo docker run --it test 

sudo docker run --privileged -it --rm --runtime nvidia --network host --device /dev/video0:/dev/video0:mrw -e DISPLAY=$DISPLAY -v /tmp/argus_socket:/tmp/argus_socket -v /tmp/.X11-unix/:/tmp/.X11-unix cargate python3 main.py --show --nano
