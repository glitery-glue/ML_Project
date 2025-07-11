import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class PredictionPipeline:
     def __init__(self):
         pass 

     def predict(self,features,target_variable):
          logging.info("inside prediction")
          try:
               if target_variable =='math_score':
                    model_path="artifacts/model_train.pkl"
                    preprocessor_path="artifacts/preprocessor.pkl"
               elif target_variable =='reading_score':
                    model_path="artifacts/model_train_1.pkl"
                    preprocessor_path="artifacts/preprocessor_1.pkl"
               else:

                    model_path="artifacts/model_train_2.pkl"
                    preprocessor_path="artifacts/preprocessor_2.pkl"
               model = load_object(model_path)
               
               scaler=load_object(preprocessor_path)
               scaled_feature= scaler.transform(features)
               predict=model.predict(scaled_feature)
               logging.info("prediction is done")
               return predict
          except Exception as e:
               raise CustomException(e,sys)
          

