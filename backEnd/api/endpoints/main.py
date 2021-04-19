from fastapi import FastAPI
import sqlite3
from crud import create, update, delete
import pandas as pd
from functions import colorID, recoginzed_plate, base64Img_cv2Img
import numpy as np
from PIL import Image
import io
import base64
import cv2
from dataModels import carImage, plateImage
import time
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
    # df = pd.read_sql_query("Select * from users", connection)
    # cp_list = df['carplate_no'].tolist()
    plates = ['WYQ8233', 'WHY1612', 'VDS2875']
    return plates

# # Update enter records
# @app.post("/enter")
# async def updateRecords(recordsDM: records):
#     update(connection, "enter_records", records=recordsDM)
#     return "Data Received!"

# # Update exit records
# @app.post("/exit")
# async def updateRecords(recordsDM: records):
#     update(connection, "exit_records", records=recordsDM)
#     return "Data Received!"

@app.post("/car")
async def carImage(inputs:carImage):
    start_time = time.time()
    carImg = base64Img_cv2Img(inputs.image)
    car_color = colorID(carImg, 2)
    print('Car Color Execution: %s seconds'% (time.time() - start_time))
    return car_color

@app.post("/plate")
async def plateImage(inputs:plateImage):
    start_time = time.time()
    plateImg = base64Img_cv2Img(inputs.image)
    plate_number = recoginzed_plate(plateImg)
    print('Plate Recognition Execution: %s seconds'% (time.time() - start_time))
    return plate_number