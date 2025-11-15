import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.ml_first_project.exception import CustomException
from src.ml_first_project.logger import logging
from src.ml_first_project.utils import save_object
import os
from urllib.parse import urlparse
import mflow
from src.ml_first_project.utils import evaluate_model

# model_trainer.py me:




@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def evel_metrics(self, actual, pred):
        rmse=np.sqrt(mean_squared_error(actual, pred))
        mae=mean_absolute_error(actual, pred)
        r2=r2_score(actual, pred)
        return rmse, mae, r2
    






    def initiate_model_trainer(self, train_array, test_array):
        try:
            
            logging.info("split training and test input data")
            X_train,y_train, X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models={
                 "Random Forest":RandomForestRegressor(),
                 "DecisionTreeRegressor":DecisionTreeRegressor(),
                 "GradientBoostingRegressor":GradientBoostingRegressor(),
                 "LinearRegression":LinearRegression(),
                 "XGBRegressor":XGBRegressor(),
                 "CatBoostRegressor":CatBoostRegressor(verbose=False),
                 "AdaBoostRegressor":AdaBoostRegressor(),
            }
            params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},

                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report:dict=evaluate_model(X_train, y_train, X_test,y_test, models, params)
            #to get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            #to get best model name from dict

           # best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            print("this is the best model:")
            print(best_model_name)



            model_names=list(params.keys())

            actual_model=" "

            for model in model_names:
                if best_model_name==model:
                    actual_model=actual_model+model
            
            best_params=params[actual_model.strip()]
            print(f"Model selected: '{actual_model}'")
            
            #os.environ['MLFLOW_TRACKING_USERNAME'] = 'imukeshkumarprajapat'
            #os.environ['MLFLOW_TRACKING_PASSWORD'] = 'f3972ad0aedfa586ce570c7fbe08534512c81809'

            mlflow.set_tracking_uri("https://dagshub.com/imukeshkumarprajapat/MLtestproject.mlflow")
            mlflow.set_experiment("Default")
            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
            

            #mlflow track
            with mlflow.start_run():
                predicted_qualities=best_model.predict(X_test)
                (rmse,mae,r2)=self.evel_metrics(y_test, predicted_qualities)
                #mlflow.log_param(best_model)
                mlflow.log_param("best_model", best_model_name.strip())  # ✅ Correct
                mlflow.log_metric("rmse",rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)
                #model registery does not work with file store
                
            if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                #mlflow.sklearn.log_model(best_model, name="model".strip())
                #mlflow.sklearn.log_model(best_model, name="model")
                mlflow.sklearn.log_model(best_model, artifact_path="model")
                
            else:
                mlflow.sklearn.log_model(best_model, "model")




            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2=r2_score(y_test, predicted)
            return r2
            

        except Exception as e:
          raise CustomException(e, sys)