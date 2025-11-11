# mysql ------> Transform ------> csv
# mysql ------> train test split--->dataset

import os
import sys
from src.ml_first_project.exception import CustomException  
from src.ml_first_project.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.ml_first_project.utils import read_sql_data

from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
       
        try:
           #reading the data from mysql database
           df=read_sql_data()
           logging.info("Reading the data from mysql database")
           os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

           df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
           train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
          
           train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
           test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

           logging.info("Data ingestion completed successfully.")

           return (self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys)