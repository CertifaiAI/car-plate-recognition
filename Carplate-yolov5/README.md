# CarGate Project Implementation using Yolov5
This following folder contains the Yolov5 project implementation which has been trained to detect 3 classes namely: Car, Van and NumberPlate. Also included is the hardware sensor scripts used to determine if a vehicle is present and also for granting access to authorized vehicles.

## Run Cargate project on PC 💻

### Run the Backend Server
Before running the project, make sure to first setup the FastAPI backend server, to do so please refer to the following [server instructions](../Kraftboard-Fastapi/README.MD).

### Install Requirements

Python 3.8 or later with all [requirements.txt](requirements.txt) dependencies installed. To install run:

```
# Clone the repo
$ git clone https://github.com/CertifaiAI/car-plate-recognition.git && cd Carplate-yolov5 

# Install requirements
$ pip install -r requirements.txt

# Download weights
$ gdown https://drive.google.com/uc?id=18tyNWkGC_x9FddZ9hJ5di3_Sc9WPkpd_ -O yolov5/weights/detection.pt
```
Note:
1. Please modify the Self.DEVICE variable to 'gpu' (if using CUDA) or 'cpu' (if using CPU) inside the file ```config.py```
2. Please refer to Pytorch [installation guide page](https://pytorch.org/get-started/locally/) for other pytorch gpu cuda versions

### Run Inference
Detect vehicles using the PC webcam and send NumberPlate detections over to a FastAPI backend server to perform LPR (no sensors involved).

```
$ python main.py --show
```  
