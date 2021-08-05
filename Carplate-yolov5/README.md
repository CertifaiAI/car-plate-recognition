# CarGate Project Implementation using Yolov5
This is the official CarGate implementation which will be cloned by the Dockerfile in the 'main' branch.

It will be used to detect vehicles including Cars and Vans and send the NumberPlate detections over to a FastAPI backend server which will perform the License Plate Recognition (LPR) and allow the access to authorized vehicles.

<br />

## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/CertifaiAI/car-plate-recognition/blob/main/Carplate-yolov5/requirements.txt) dependencies installed. To install run:

```bash
# Download repo
$ git clone https://github.com/CertifaiAI/car-plate-recognition.git && cd Carplate-yolov5 

# Install thrid party lib
$ pip install -r requirements.txt

# Download weights
$ gdown https://drive.google.com/uc?id=18tyNWkGC_x9FddZ9hJ5di3_Sc9WPkpd_ -O yolov5/weights/detection.pt
```
Notes:
1. Please modify the Settings.DEVICE variable to cuda (Using GPU) or cpu (Using cpu) under the file Core/config.py
2. Please refer to Pytorch [installation guide page](https://pytorch.org/get-started/locally/) for other pytorch gpu cuda version

<br />

## Run the Server

Before running the scripts make sure to first setup the FastAPI backend server, to do so please refer to the following [instructions](https://github.com/CertifaiAI/car-plate-recognition/blob/main/Backend-server/README.MD).

<br />

## Inference
*main.py*: Runs the cargate program without sensors.
```bash
$ python main.py --show
```  

*send_images.py*: Iterates through a number of pre-saved NumberPlate images, sends the images over to the FastAPI backend server and outputs the LPR. 

```bash
$ python send_images.py 
```

*send_video.py*: Detects vehicles including Cars and Vans and sends cropped images of the NumberPlate over to a FastAPI backend server which performs the LPR. 

 ```bash
$ python send_video.py
```

