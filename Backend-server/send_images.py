import cv2
import requests
import base64
import json
from PIL import Image
import io
import os

# Covert cv2 images to base64 images
def cv2Img_base64Img(cv2Img):
    # array to Pil
    image = Image.fromarray(cv2Img)
    # Pil to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('ascii')


if __name__== "__main__":
    # loop through a directory of images 
    # dirs = os.listdir('images') # Set directory here
    # for file in dirs:
    #     print(file)
    #     # Insert iamge here
    #     carImage = cv2.imread("images/"+file)
    #     # Convert image to correct color 
    #     carImage = cv2.cvtColor(carImage, cv2.COLOR_BGR2RGB)
    #     # convert image to base64
    #     carImage = cv2Img_base64Img(carImage)
    #     data = {"image": carImage}
    #     # Send image to server
    #     response = requests.post('http://localhost:8000/plate', data=json.dumps(data))
    #     print(response.text)
    
    carImage = cv2.imread("images/FORTUNER.jpg")
    # Convert image to correct color 
    carImage = cv2.cvtColor(carImage, cv2.COLOR_BGR2RGB)
    # convert image to base64
    carImage = cv2Img_base64Img(carImage)
    data = {"image": carImage}
    # Send image to server
    response = requests.post('http://localhost:8000/plate', data=json.dumps(data))
    print(response.text)