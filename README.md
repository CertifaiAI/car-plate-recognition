# Cargate Project Implementation
#### The system is built using yolov5 (objects detector) [Reference 1] and PaddlePaddle OCR (license plate recognition) [Reference 2]

It will be used to detect vehicles including Cars and Vans and send the NumberPlate detections over to a FastAPI backend server which will perform the License Plate Recognition (LPR) and allow the access to authorized vehicles.

#### Run the Backend Server
Before running the project, make sure to first setup the FastAPI backend server, to do so please refer to the following [instructions](Kraftboard-Fastapi/README.MD).

#### This project can be run on both a PC 💻 as well as on a Jetson Nano 📼    
- To Run the Cargate project on a PC, refer to the following [instructions](Carplate-yolov5/README.md).
- To Run the Cargate project on a Jetson Nano, choose from the following two options:   
  - [Run Cargate on Jetson Nano without Docker](#run-cargate-on-jetson-nano-without-docker)
  - [Run Cargate on Jetson Nano with Docker](#run-cargate-on-jetson-nano-with-docker)

<br />

## Run Cargate on Jetson Nano without Docker
The Following scripts were tested on a Jetson Nano with **JetPack 4.5** image flashed. To download Jetpack 4.5, refer to the following [link](https://developer.nvidia.com/jetpack-sdk-45-archive).

### Install torch and torchvision libraries
```
./install_torch.sh
```

### Install additional libraries
```
pip3 install -r Carplate-yolov5/requirements-nano.txt
```

### Download weights
```
gdown https://drive.google.com/uc?id=18tyNWkGC_x9FddZ9hJ5di3_Sc9WPkpd_ -O Carplate-yolov5/yolov5/weights/detection.pt
```

### Run Inference
Detect vehicles using the CSI Camera and send NumberPlate detections over to a FastAPI backend server to perform LPR.
```
python3 Carplate-yolov5/main.py --show --nano
```

<br />

## Run Cargate on Jetson Nano with Docker
The Following scripts were tested on a Jetson Nano with **JetPack 4.5** image flashed. To download Jetpack 4.5, refer to the following [link](https://developer.nvidia.com/jetpack-sdk-45-archive).

### Change Docker Default Runtime
This sets ```"default-runtime": "nvidia"``` in ```/etc/docker/daemon.json```
```
./setup.sh
```

### Build the Docker Image 
Note: This should take approximately 3-4 hours to complete
```
./build.sh
```

### Run Inference
Detect vehicles using the CSI Camera and send NumberPlate detections over to a FastAPI backend server to perform LPR.
```
./run.sh
```

<br />

## **Demo**
![car and license plate detection](./result.jpg)

<br />

## External Hardware Circuit Diagram
There are 2 external sensors used in this project:
1) **HC-SR04 Ultrasonic Distance Sensor**: Used to determine if a vehicle is present before starting the detection. [(link)](https://my.cytron.io/p-3v-5.5v-ultrasonic-ranging-module?r=1&gclid=Cj0KCQjw3f6HBhDHARIsAD_i3D-F-Jbar0A6EUIiDZ_Uve30oZ26GYebag4zr8nsH9GjRjN0Baa66QMaAm-wEALw_wcB)          
2) **1 CH Active H/L 5V Relay Module**: Used to control the gate and grant access to authorized vehicles. [(link)](https://my.cytron.io/p-1ch-active-h-l-5v-optocoupler-relay-module?search=single%20channel&description=1)       

![CarGate Circuit Diagram](https://user-images.githubusercontent.com/68045710/127173320-dfba41cb-7f76-4e7c-98df-4b95899fe72c.PNG)

## Appendix

There are 2 ways that you can use to run shell scripts on Linux:
```
1) By using the bash command:   
$ bash ./scriptname.sh   

2) By making the script executable (To be performed only once):        
$ chmod +x scriptname.sh   

Then run the script normally using the command:   
$ ./scriptname.sh      
```

<br />

## **References**
#### 1. https://github.com/ultralytics/yolov5
#### 2. https://github.com/PaddlePaddle/PaddleOCR 
#### 3. https://github.com/otamajakusi/dockerfile-yolov5-jetson
