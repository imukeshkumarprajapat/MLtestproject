from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import sys
import dagshub

from src.ml_first_project.logger import logging
from src.ml_first_project.exception import CustomException

from src.ml_first_project.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.ml_first_project.components.data_transformation import DataTransformation, DataTransformationConfig
from src.ml_first_project.components.model_trainer import ModelTrainer, ModelTrainerConfig

from src.ml_first_project.pipelines.prediction_pipeline import CustomData, PredictionPipline


from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

dagshub.init(repo_owner='imukeshkumarprajapat', repo_name='MLtestproject', mlflow=True)

if __name__ == "__main__":
    logging.info("Starting the ML First Project application...")

    try:
        # Simulate some code that may raise an exception
       # Data_ingestion_config = DataIngestionConfig()
       data_ingestion= DataIngestion()
       train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()
       #data_transformation_config=DataTransformationConfig()
       data_transformation=DataTransformation()

       train_arr, test_arr,preprocessor_path=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

       #model Training 

       model_trainer=ModelTrainer()
       print(model_trainer.initiate_model_trainer(train_arr, test_arr))


    except Exception as e:
        logging.info("custom exception raised")
        raise CustomException(e, sys)
    
    application=Flask(__name__)
    app=application

    #route for a home page

    @app.route('/')
    def index():
        
        return render_template("index.html")
    


    @app.route("/predictdata",methods=['GET','POST'])
    def predict_datapoint():
       if request.method=="GET":
           return render_template("home.html")
       else:
           data=CustomData(
           
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

           pred_df=data.get_data_as_data_frame()
           print(pred_df)
           print("Before Prediction")

           predict_pipeline=PredictionPipline()
           print("Mid Prediction")
           results=predict_pipeline.predict(pred_df)
           print("after Prediction")
           return render_template('home.html',results=results[0])


if __name__=="__main__":
    app.run(host="0.0.0.0",  debug=True)


