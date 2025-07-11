import os
import sys
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException

def save_obj(file_path, obj):
    try:

        obj_dir=os.path.dirname(file_path)
        os.makedirs(obj_dir, exist_ok=True)
        with open (file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def save_df(file_path, df):
    try:
        obj_dir=os.path.dirname(file_path)
        os.makedirs(obj_dir, exist_ok=True)
        df.to_csv(file_path)
    except Exception as e:
        raise CustomException(e, sys)   

def evaluate_model(X_train, y_train, X_test, y_test, models:dict, param:dict):
    logging.info("inside evaluate model")
    try:
        report={}
        best_params_dict={}
        for i in range (len(list(models))):
            model=list(models.values())[i]
            model_name=list(models.keys())[i]
            para=param[model_name]
            grid_search=GridSearchCV(model,para,cv=3)
            grid_search.fit(X_train,y_train)
            best_params=grid_search.best_params_
            best_params_dict[model_name] = best_params
            model.set_params(**grid_search.best_params_)

            model.fit(X_train, y_train)
            logging.info("hyperparameter tuning is done")
            #prediction
            y_predict_test = model.predict(X_test)
            y_predict_train=model.predict(X_train)
            logging.info("got the prediction")
            train_model_score = r2_score(y_train, y_predict_train)
            test_model_score = r2_score(y_test, y_predict_test)
            logging.info("got the r2 ")
            report[model_name]=test_model_score
        return report,best_params_dict
    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)