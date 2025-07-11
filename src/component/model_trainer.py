import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model,save_obj
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


DATA_DIR = 'artifacts'
os.makedirs(DATA_DIR,exist_ok=True)

@dataclass
class ModelTrainerConfig:
    result_path = os.path.join(DATA_DIR,'report_2.csv')
    param_path =os.path.join(DATA_DIR,"best_params_2.csv")
    model_train_path = os.path.join(DATA_DIR,'model_train_2.pkl')
    model_param_path= os.path.join(DATA_DIR,'model_param_2.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,train_array, test_array):
        logging.info("Model training has initiated")
        try:
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models= {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Ridge Regression": {
                     'alpha': [0.1, 1.0, 10.0, 100.0],
                     'fit_intercept': [True, False]
                },
                "Lasso Regression": {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'fit_intercept': [True, False]
                }
                
            }

            model_report,best_param=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            model_score_df = pd.DataFrame(list(model_report.items()), columns=['Model_Name', 'R_2_score']).sort_values(by='R_2_score', ascending=False)
            model_score_df.to_csv(self.model_trainer_config.result_path)
            best_param_df = pd.DataFrame(list(best_param.items()),columns=['model_name','best_param'])
            best_param_df.to_csv(self.model_trainer_config.param_path)
            logging.info(f"Model training completed. Results saved to {self.model_trainer_config.result_path}")
            # get best model
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            best_model_param = best_param[best_model_name]
        
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            # saving best tuned model
            best_model.set_params(**best_model_param)
            best_model.fit(X_train, y_train)
            save_obj(file_path=self.model_trainer_config.model_train_path,obj=best_model)
            
            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)       