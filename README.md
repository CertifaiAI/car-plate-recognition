# Cargate Project Implementation on Jetson Nano
#### The system is built using yolov5 (objects detector) [Reference 1] and PaddlePaddle OCR (license plate recognition) [Reference 2]

The Following scripts work on a Jetson Nano with **JetPack 4.5** image flashed. To download Jetpack 4.5, refer to the following [link](https://developer.nvidia.com/jetpack-sdk-45-archive).

## Steps to Run
### Change Docker Default Runtime
This sets ```"default-runtime": "nvidia"``` in ```/etc/docker/daemon.json```
```
./setup.sh
```
### Build the Docker Image 
Note: This should take approximately 3hours to complete
```
./build.sh
```
### Check Camera is Detected
Check to insure that the webcam is connected to /dev/video0
```
ls -ltrh /dev/video*
```
### Download Weights

Due to the Github memory limit, the weights have been uploaded on to [Google Drive](https://drive.google.com/drive/folders/1afPFDv9Fo0GW4W5ss6GWgBGX31iUmn4t), please the download the weights and place it inside of a folder named 'weights/' on your Jetson Nano, then modify the --volume command in ```run.sh``` to point to that folder.

### Inference
Detect vehicles using CSI Camera connected to a Jetson Nano and send NumberPlate detections over to a FastAPI backend server to perform LPR
```
./run.sh
```


### **Demo**
![car and license plate detection](./result.jpg)
#

### **References**
#### 1. https://github.com/ultralytics/yolov5
#### 2. https://github.com/PaddlePaddle/PaddleOCR
#### 3. https://github.com/otamajakusi/dockerfile-yolov5-jetson
