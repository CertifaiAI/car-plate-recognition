# CarGate Project Implementation using Yolov5
This is the official CarGate implementation which will be cloned in the 'main' branch Dockerfile.

It will be used to detect vehicles including Cars and Vans and send the NumberPlate detections over to a FastAPI backend server which will perform the License Plate Recognition (LPR) and allow the access to authorized vehicles.

### Requirements

Python 3.8 or later with all [requirements.txt](requirements.txt) dependencies installed. To install run:

$ git clone --branch Carplate-yolov5 https://github.com/CertifaiAI/car-plate-recognition.git  
$ pip install -r requirements.txt

### Download Weights

Due to the Github memory limit, the weights have been uploaded on to [Google Drive](https://drive.google.com/drive/folders/1afPFDv9Fo0GW4W5ss6GWgBGX31iUmn4t), please the download the weights and place it inside of the 'weights/' folder

### Run Server

Before testing the scripts make sure to first setup the FastAPI backend server, to do so please refer to the following instructions.

### Inference

*send_images.py*: Iterates through a number of pre-saved NumberPlate images, sends the images to the FastAPI server and outputs the LPR. 

```bash
$ python send_images.py 
```

*send_video.py*: Detects vehicles including Cars and Vans and sends a cropped image of the NumberPlate onto the FastAPI server to perform LPR. 

 ```bash
$ python send_video.py
```
