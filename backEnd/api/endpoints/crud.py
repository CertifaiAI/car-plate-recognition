import sqlite3
import pandas as pd

# test
connection = sqlite3.connect("ALNP.db")

def create(connection):
    try:
        cursor = connection.cursor()
        # Create Users
        create_user_query = 'create table users(carplate_no char(20) PRIMARY KEY, emp_ID char(20), name char(50));'
        cursor.execute(create_user_query)
        print("Users table created")
        # Create enter records
        create_enter_records_query = 'create table enter_records(carplate_no char(20), In_time text);'
        cursor.execute(create_enter_records_query)
        print("Enter Records table created")
        # Create exit records
        create_exit_records_query = 'create table exit_records(carplate_no char(20), Out_time text);'
        cursor.execute(create_exit_records_query)
        print("Exit Records table created")
    except connection.Error as err:
        print("Cannot connect to database ", err)

def delete(carplate_no: str, db_name: str, connection):
    try:
        delete_query = f'Delete from {db_name} where carplate_no={repr(carplate_no)};'
        cursor = connection.cursor()
        cursor.execute(delete_query)
        connection.commit()
    except connection.Error as err:
        print("Unable to delete", err)


def update(connection, db_name: str, records = " ", users = " "):
    try:
        if db_name == "users":      
            # Check PK
            df = pd.read_sql_query("Select * from users", connection)
            cp_list = df['carplate_no'].tolist()
            # print(cp_list)
            # Remove old 
            if users.carplate_no in cp_list:
                delete(users.carplate_no, db_name, connection)
                # Insert new
            insert_query = 'Insert into {} (emp_id, name, carplate_no) values ({}, {}, {})'.format(db_name, repr(users.emp_id), repr(users.name), repr(users.carplate_no))
            cursor = connection.cursor()
            cursor.execute(insert_query)
            # Comfirm changes
            connection.commit()
            print("Records Updated")
        elif db_name == "enter_records":
            insert_query = 'Insert into {} values ({}, {})'.format(db_name, repr(records.carplate_no), repr(records.In_time))
            cursor = connection.cursor()
            cursor.execute(insert_query)
            # Comfirm changes
            connection.commit()
        elif db_name == "exit_records":
            insert_query = 'Insert into {} values ({}, {})'.format(db_name, repr(records.carplate_no), repr(records.Out_time))
            cursor = connection.cursor()
            cursor.execute(insert_query)
            # Comfirm changes
            connection.commit()
    except connection.Error as err:
        print("Cannot connect to database", err)

def select_all(connection):
    try:
        df = pd.read_sql_query("Select * from records", connection)
        print(df)
        df = pd.read_sql_query("Select * from users", connection)
        print(df)
    except connection.Error as err:
        print("Cannot connect to database ", err)

# create(connection)