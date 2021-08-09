import cv2
import requests
import base64
import json
from PIL import Image
import io

import glob

def cv2Img_base64Img(cv2Img):
    # array to Pil
    image = Image.fromarray(cv2Img)
    # Pil to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('ascii')


directory = 'data/images/*'

all_access = ['WND1288', 'WMJ8663']

for carPlate in glob.glob(directory):
    carImage = cv2.imread(carPlate)
    carImage = cv2.cvtColor(carImage, cv2.COLOR_BGR2RGB)
    carImage = cv2Img_base64Img(carImage)
    data = {"image": carImage}
    response = requests.post('http://localhost:8000/api/ocr/ocr', data=json.dumps(data))
    print(response.text)
    if json.loads(response.text)['Plate Number'] in all_access:
        print("Activated! Welcome " + json.loads(response.text)['Plate Number'])
