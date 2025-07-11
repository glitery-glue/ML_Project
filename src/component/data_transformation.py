import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import save_obj
from src.logger import logging
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_DIR = r'student_performance\artifacts'
os.makedirs(DATA_DIR,exist_ok=True)
@dataclass
class DataTransformationConfig:
    processor_obj_file_path = os.path.join(DATA_DIR, 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transform_config=DataTransformationConfig

    def get_data_transformation_pipeline(self, input_df):
        ''' This function is to preparer the transformation pipeline as preprocessing'''
        try:
            
            numeric_feature=input_df.select_dtypes(exclude=object).columns
            categorical_feature=input_df.select_dtypes(include=object).columns
            
            num_pipeline=Pipeline(
                steps=
                [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            char_pipeline=Pipeline(
                steps=
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))

                ])
            
            logging.info(f"Categorical columns: {categorical_feature}")
            logging.info(f"Numerical columns: {numeric_feature}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numeric_feature),
                    ("char_pipeline",char_pipeline,categorical_feature)
                ]
            )
            return preprocessor
        except Exception as e:
            raise (CustomException(e,sys))


    def transform_data(self, train_path, test_path, target_variable):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("train and test data set is loaded for pre processing")
            input_feature_train_df=train_df.drop(columns=[target_variable],axis=1)
            target_feature_train_df=train_df[target_variable]
            input_feature_test_df=test_df.drop(columns=[target_variable],axis=1)
            target_feature_test_df=test_df[target_variable]
            preprocessing_obj=self.get_data_transformation_pipeline(input_feature_train_df)
            
            logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe."
                )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_array=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_array=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("saving preprocessing object")
            save_obj(self.data_transform_config.processor_obj_file_path, preprocessing_obj)

            return(train_array, test_array,self.data_transform_config.processor_obj_file_path)
        except Exception as e:
            raise CustomException(e, sys)


if __name__=="__main__":
    target_variable='math_score'
    train_path = r"student_performance\src\component\artifacts\train.csv"
    test_path= r"student_performance\src\component\artifacts\test.csv"
    data_transformation = DataTransformation()
    _,_,processor_path=data_transformation.transform_data(train_path,test_path,target_variable)
    print(processor_path)