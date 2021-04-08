import cv2
import requests
import base64
import json
from PIL import Image
import io
import numpy as np

def cv2Img_base64Img(cv2Img):
    # array to Pil
    image = Image.fromarray(cv2Img)
    # Pil to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('ascii')




image = cv2.imread('/home/nelson/Desktop/saved_detections/PFQ5217.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
test = cv2Img_base64Img(image)
# print(test)
# retval, buffer = cv2.imencode('.jpg', image)
# print(buffer)
# jpg_as_text = base64.b64encode(buffer)
# print(jpg_as_text)
data = {'image': test}
try:
    response = requests.post('http://10.10.10.35:8000/plate', data=json.dumps(data))
except:
    print('failed')
print(str(response.text))
