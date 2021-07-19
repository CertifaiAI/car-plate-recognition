from fastapi import FastAPI
from functions import recoginzed_plate, base64Img_cv2Img
import time
from pydantic import BaseModel

class plateImage(BaseModel):
    image: str

# FastAPI
app = FastAPI()

# PPOCR Here
@app.get("/")
def home():
    return{"message": "Hello World"}

@app.post("/plate")
async def plateImage(inputs:plateImage):
    # Get execution time
    start_time = time.time()
    # Run PPOCR here
    results = recoginzed_plate(base64Img_cv2Img(inputs.image))
    execution_time = time.time() - start_time
    # Set total execution time to result
    results['Execution time'] = round(float(execution_time),2)
    return results