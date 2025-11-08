from src.ml_first_project.logger import logging
from src.ml_first_project.exception import CustomException
import sys
from src.ml_first_project.components.data_ingestion import DataIngestion
from src.ml_first_project.components.data_ingestion import DataIngestionConfig

if __name__ == "__main__":
    logging.info("Starting the ML First Project application...")

    try:
        # Simulate some code that may raise an exception
        #Data_ingestion_config = DataIngestionConfig()
        data_ingestion= DataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.info("custom exception raised")
        raise CustomException(e, sys)