import cv2
import requests
import base64
import json
from PIL import Image
import io
import numpy as np
from datetime import datetime, timedelta

def cv2Img_base64Img(cv2Img):
    # array to Pil
    image = Image.fromarray(cv2Img)
    # Pil to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('ascii')

carImage = cv2.imread('/home/nexlson/Skymind/Cargate/car-plate-recognition/src/croppedCar.jpg')
# plateImage = cv2.imread('/home/nexlson/Skymind/Cargate/car-plate-recognition/src/croppedPlate.jpg')
carImage = cv2.cvtColor(carImage, cv2.COLOR_BGR2RGB)
# plateImage = cv2.cvtColor(plateImage, cv2.COLOR_BGR2RGB)
carImage = cv2Img_base64Img(carImage)
# plateImage = cv2Img_base64Img(plateImage)
# entryTime = datetime.now()
# exitTime = datetime.now() + timedelta(hours=3)
# data = {'plate_number': 'PLA6626','carImage': carImage, 'entry_time':str(entryTime)}
# data = {'plate_number': '','carImage': carImage, 'entertime':'', 'exittime':''}
entryTime = '2021-04-06 09:32:29'
exitTime =  ''
plate_number = 'PLA6626'
auth = 'True'


data = {'plate_number': plate_number,'carImage': carImage, 'entertime':str(entryTime), 'exittime':str(exitTime), 'Authenticated':auth}
try:
    response = requests.post('http://localhost:8080/api/v1/tngSC38jlSwK7icJV2j0/telemetry', data=json.dumps(data))
except:
    print('failed')
print((response.text))
