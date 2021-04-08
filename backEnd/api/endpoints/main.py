from fastapi import FastAPI
import sqlite3
from crud import create, update, delete
import pandas as pd
from models.enter_records import enter_Records
from models.exit_records import exit_Records
from models.users import Users
from models.ai_process import ai_process
from aiFunctions import colorID, recognizePlate
import numpy as np

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


@app.post("/ai")
async def aiProcess(inputs: ai_process):
    print(inputs)
    # convert img back 
    carImage = np.frombuffer(base64.b64decode(inputs.carImg)).reshape(inputs.carShape)
    plateImage = np.frombuffer(base64.b64decode(inputs.plateImg)).reshape(inputs.plateShape)
    plate_number = recognizePlate(plate)
    car_color = colorID(car)
    return plate_number, car_color