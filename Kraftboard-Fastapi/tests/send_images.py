import cv2
import requests
import base64
import json
from PIL import Image
import io
import numpy as np
import pickle

# Covert cv2 images to base64 images
def cv2Img_base64Img(cv2Img):
    # array to Pil
    image = Image.fromarray(cv2Img)
    # Pil to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('ascii')

def array2list(npArray):
    return (npArray.tolist())

def list2array(list):
    return (np.array(list))

def array_to_base64(array):
    bytes_array = pickle.dumps(array)
    array_base64 = base64.b64encode(bytes_array).decode('ascii')
    return array_base64


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
    
    carImage = cv2.imread("face.jpg")

    points = [[[164.71205139160156, 464.80303955078125], [399.141845703125, 460.17681884765625], [282.9638366699219, 618.928466796875], [174.51646423339844, 708.09716796875], [402.64697265625, 708.177490234375]]]
    boxes = [[26.444469451904297, 185.99990844726562, 545.4476928710938, 882.7085571289062]]

    points = array_to_base64(list2array(points))
    boxes = array_to_base64(list2array(boxes))

    print(points)
    print(boxes)

    # Convert image to correct color 
    carImage = cv2.cvtColor(carImage, cv2.COLOR_BGR2RGB)
    # convert image to base64
    carImage = cv2Img_base64Img(carImage)
    # data = {"image": carImage, 'boxes': boxes, 'points': points}
    data = {"image": carImage}


    # Send image to server
    response = requests.post('http://localhost:8000/api/face/mtcnnEmbeddings', data=json.dumps(data))
    print(response.text)