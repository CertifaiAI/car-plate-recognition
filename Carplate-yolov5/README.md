# CarGate Project Implementation using Yolov5
This is the official CarGate implementation which will be cloned by the Dockerfile in the 'main' branch.

It will be used to detect vehicles including Cars and Vans and send the NumberPlate detections over to a FastAPI backend server which will perform the License Plate Recognition (LPR) and allow the access to authorized vehicles.

## Steps to Run on a PC
* Update: After the latest PR Allowing access to GPIO pins on the Jetson Nano for controlling an external Relay and Ultasonic Distance Sensor, the following steps can no longer be used to test on a PC.
### Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/CertifaiAI/car-plate-recognition/blob/main/Carplate-yolov5/requirements.txt) dependencies installed. To install and download weights run:

```bash
$ git clone https://github.com/CertifaiAI/car-plate-recognition.git
$ cd Carplate-yolov5
$ pip install -r requirements.txt
$ gdown https://drive.google.com/uc?id=18tyNWkGC_x9FddZ9hJ5di3_Sc9WPkpd_ -O weights/detection.pt
```

### Download Weights

Due to the Github memory limit, the weights have been uploaded on to [Google Drive](https://drive.google.com/drive/folders/1afPFDv9Fo0GW4W5ss6GWgBGX31iUmn4t), please download the weights and place it inside of the 'weights/' folder in this project directory.

### Run the Server

Before running the scripts make sure to first setup the FastAPI backend server, to do so please refer to the following [instructions](https://github.com/CertifaiAI/car-plate-recognition/blob/main/Backend-server/README.MD).

### Inference

*send_images.py*: Iterates through a number of pre-saved NumberPlate images, sends the images over to the FastAPI backend server and outputs the LPR. 

```bash
$ python send_images.py 
```

*send_video.py*: Detects vehicles including Cars and Vans and sends cropped images of the NumberPlate over to a FastAPI backend server which performs the LPR. 

 ```bash
$ python send_video.py
```

