
import os
import sys
from src.ml_first_project.exception import CustomException  
from src.ml_first_project.logger import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pymysql

load_dotenv()


host = os.getenv("host")
user = os.getenv("user")
password= os.getenv("password")
db= os.getenv("db")




def read_sql_data():
    logging.info("Establishing connection to the MySQL database")
    try:
       mydb=pymysql.connect(
              host=host,
              user=user,
              password=password,
              database=db
         )
       logging.info("Connection established successfully",mydb)
       df=pd.read_sql_query("SELECT * FROM students",mydb)
       print(df.head())
       logging.info("Data read successfully from the database")
       return df
    
    except Exception as e:
        raise CustomException(e)