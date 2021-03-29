from fastapi import FastAPI
import sqlite3
from crud import create, update, delete
import pandas as pd
from models.enter_records import enter_Records
from models.exit_records import exit_Records
from models.users import Users


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
async def updateRecords(recordsDM: enter_Records):
    update(connection, "enter_records", records=recordsDM)
    return "Data Received!"

# Update exit records
@app.post("/exit")
async def updateRecords(recordsDM: exit_Records):
    update(connection, "exit_records", records=recordsDM)
    return "Data Received!"

# FrontEnd
# Get data from users
@app.get("/users")
async def getUsers():
    df = pd.read_sql_query("Select * from users", connection)
    return df

# Get data from enter_records
@app.get("/enter")
async def getRecords():
    df = pd.read_sql_query("Select * from enter_records", connection)
    return df

# Get data from exit_records
@app.get("/exit")
async def getRecords():
    df = pd.read_sql_query("Select * from exit_records", connection)
    return df

# Update or insert new users
@app.post("/users")
async def updateUsers(usersDM: Users):
    update(connection, "users", users=usersDM)
    return "Received"

@app.delete("/users")
async def deleteUsers(carplate_no: str):
    delete(carplate_no, "users", connection)
    return "Data Deleted"
