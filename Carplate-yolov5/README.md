# CarGate Project Implementation using Yolov5
This is the official CarGate implementation which will be cloned by the Dockerfile in the 'main' branch.

It will be used to detect vehicles including Cars and Vans and send the NumberPlate detections over to a FastAPI backend server which will perform the License Plate Recognition (LPR) and allow the access to authorized vehicles.

## Steps to Run on a PC
### Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/CertifaiAI/car-plate-recognition/blob/main/Carplate-yolov5/requirements.txt) dependencies installed. To install run:

```bash
$ git clone https://github.com/CertifaiAI/car-plate-recognition.git
$ cd Carplate-yolov5
$ pip install -r requirements.txt
$ gdown https://drive.google.com/uc?id=18tyNWkGC_x9FddZ9hJ5di3_Sc9WPkpd_ -O yolov5/weights/detection.pt
```

### Run the Server

Before running the scripts make sure to first setup the FastAPI backend server, to do so please refer to the following [instructions](https://github.com/CertifaiAI/car-plate-recognition/blob/main/Backend-server/README.MD).

### Inference
*main.py*: Runs the cargate program as a whole (includes sensors). User can select which sensors to use
```bash
$ python main.py --show
```  
This show output result on PC.

```bash
$ python main.py --show --nano
```  
This show output result on Jetson Nano.

```bash
$ python main.py --show --nano --sensor --led --relay
```  
This show output result on Jetson Nano with all sensor.


*send_images.py*: Iterates through a number of pre-saved NumberPlate images, sends the images over to the FastAPI backend server and outputs the LPR. 

```bash
$ python send_images.py 
```

*send_video.py*: Detects vehicles including Cars and Vans and sends cropped images of the NumberPlate over to a FastAPI backend server which performs the LPR. 

 ```bash
$ python send_video.py
```

### Run with GPU
In order to run using GPU, modify device on Carplate-yolov5/config.py
```
self.DEVICE = 'cpu'
```
