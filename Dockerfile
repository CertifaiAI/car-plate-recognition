# Switch to 1.7
FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y cmake libgtk2.0-dev wget
# ffmpeg (CSI Camera)
RUN apt install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavresample3
# gstreamer (CSI Camera)
RUN apt install -y libgstreamer-opencv1.0-0 libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
# Install opencv
RUN git clone --recursive https://github.com/skvark/opencv-python.git
RUN python3 -m pip install --upgrade pip
RUN cd opencv-python && python3 -m pip wheel . --verbose && find . -name "opencv_python*.whl" | xargs python3 -m pip install

# Copy folder to docker 
COPY ./Carplate-yolov5 /app
RUN cd/app/Carplate-yolov5 && python3 -m pip install -r requirements-nano.txt

WORKDIR /app/Carplate-yolov5