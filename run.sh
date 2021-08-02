#!/bin/bash
xhost +local:
sudo docker run --privileged -it --rm --runtime nvidia --network host --device /dev/video0:/dev/video0:mrw -e DISPLAY=$DISPLAY -v /tmp/argus_socket:/tmp/argus_socket -v /tmp/.X11-unix/:/tmp/.X11-unix cargate python3 main.py --show --nano
