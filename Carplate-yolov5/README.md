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
```

### Download Weights

Due to the Github memory limit, the weights have been uploaded on to [Google Drive](https://drive.google.com/drive/folders/1afPFDv9Fo0GW4W5ss6GWgBGX31iUmn4t), please download the weights and place it inside of the 'weights/' folder in this project directory.

### Run the Server

Before running the scripts make sure to first setup the FastAPI backend server, to do so please refer to the following instructions.

### Inference

*send_images.py*: Iterates through a number of pre-saved NumberPlate images, sends the images over to the FastAPI backend server and outputs the LPR. 

```bash
$ python send_images.py 
```

*send_video.py*: Detects vehicles including Cars and Vans and sends the cropped images of the NumberPlate over to a FastAPI backend server to perform LPR. 

 ```bash
$ python send_video.py
```

