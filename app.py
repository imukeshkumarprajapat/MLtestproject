from src.ml_first_project.logger import logging
from src.ml_first_project.exception import CustomException
import sys
from src.ml_first_project.components.data_ingestion import DataIngestion
from src.ml_first_project.components.data_ingestion import DataIngestionConfig
from src.ml_first_project.components.data_transformation import DataTransformationConfig, DataTransformation

if __name__ == "__main__":
    logging.info("Starting the ML First Project application...")

    try:
        # Simulate some code that may raise an exception
       # Data_ingestion_config = DataIngestionConfig()
       data_ingestion= DataIngestion()
       train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()
       #data_transformation_config=DataTransformationConfig()
       data_transformation=DataTransformation()
       data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    except Exception as e:
        logging.info("custom exception raised")
        raise CustomException(e, sys)