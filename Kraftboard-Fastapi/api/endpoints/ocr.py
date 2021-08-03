from fastapi import APIRouter
from models.ocr import plateImage
import time
from utils.ocr.functions import recoginzed_plate, base64Img_cv2Img

rounter = APIRouter()

@rounter.get('/')
def helloWorld():
    return 'API for PaddlePaddle OCR'

# Recive base64Img return plate number
@rounter.post('/ocr')
async def plateImage(inputs:plateImage):
    # Get execution time
    start_time = time.time()
    # Run PPOCR here
    results = recoginzed_plate(base64Img_cv2Img(inputs.image))
    execution_time = time.time() - start_time
    # Set total execution time to result
    results['Execution time'] = round(float(execution_time),2)
    return results