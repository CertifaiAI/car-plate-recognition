from fastapi import FastAPI
import sqlite3
from crud import create, update, delete
import pandas as pd
from models.records import records
from models.ai_process import ai_process
from models.image import image
from aiFunctions import colorID, recognizePlate
import numpy as np
from PIL import Image
import io
import base64
import cv2
from ppocr.utility import parse_args
from ppocr.predict_system import TextSystem

# Connect db + startup
try:
    connection = sqlite3.connect("ALNP.db")
    # Tables list 
    df = pd.read_sql_query("Select name from sqlite_master where type='table'", connection)
    table_list = df['name'].tolist()
    # Create table if not exist
    if 'users' not in table_list and 'enter_records' not in table_list and 'exit_records' not in table_list:
        create(connection)
except connection.Error as error:
    print("Error connecting to sqlite", error)

# FastAPI
app = FastAPI()

# Nano
# Get carplates data
@app.get("/carplates")
async def getCarPlates():
    # list_of_carplate = return_carplate_list(connection)
    # print(list_of_carplate)
    df = pd.read_sql_query("Select * from users", connection)
    cp_list = df['carplate_no'].tolist()
    return cp_list

# Update enter records
@app.post("/enter")
async def updateRecords(recordsDM: records):
    update(connection, "enter_records", records=recordsDM)
    return "Data Received!"

# Update exit records
@app.post("/exit")
async def updateRecords(recordsDM: records):
    update(connection, "exit_records", records=recordsDM)
    return "Data Received!"


# @app.post("/ai")
# async def aiProcess(inputs: ai_process):
#     print(inputs)
#     # convert img back 
#     carImg = base64Img_cv2Img(inputs.carImg)
#     # convert plate back
#     plateImg = base64Img_cv2Img(inputs.plateImg) 
#     plate_number = recognizePlate(plateImg)
#     car_color = colorID(carImg)
#     return plate_number, car_color

@app.post("/car")
async def carImage(inputs:image):
    # print(inputs.image)
    carImg = base64Img_cv2Img(inputs.image)
    car_color = colorID(carImg, 3)
    return car_color

@app.post("/plate")
async def plateImage(inputs:image):
    # print(inputs.image)
    plateImg = base64Img_cv2Img(inputs.image)
    # initialize system
    sys = TextSystem(parse_args)
    # recognize plate
    dt_boxes, rec_res = sys(plateImg)
    print(rec_res)
    for text, score in rec_res:
            print("{}, {:.3f}".format(text, score))
    return plate_number