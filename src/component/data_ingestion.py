import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer

DATA_DIR = 'artifacts'
os.makedirs(DATA_DIR,exist_ok=True)
@dataclass
class DataIngetionConfig:
    train_path=os.path.join(DATA_DIR, "train.csv")
    test_path=os.path.join(DATA_DIR,'test.csv')
    raw_path=os.path.join(DATA_DIR,'data.csv')

class DataIngetion:
    def __init__(self):
        self.data_ingetion_config=DataIngetionConfig()

    def initiate_data_ingition(self, file_path):
        logging.info('Entered the data ingetion method or componet')
        try:
            df=pd.read_csv(file_path)
            
            df.to_csv(self.data_ingetion_config.raw_path, index=False,header=True )
            logging.info("raw data got saved to the data.csv")
            train_df, test_df = train_test_split(df,test_size=0.2, random_state=42)
            logging.info("train test splitting is done")

            train_df.to_csv(self.data_ingetion_config.train_path, index=False, header=True)
            test_df.to_csv(self.data_ingetion_config.test_path, index=False, header=True)
            return(
                self.data_ingetion_config.train_path,
                self.data_ingetion_config.test_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__=="__main__":
    obj=DataIngetion()
    file_path=r"artifacts\data\student.csv"
    train_data_path, test_data_path = obj.initiate_data_ingition(file_path)
    transform= DataTransformation()
    target_variable='writing_score'
    train_array, test_array,_ = transform.transform_data(train_data_path, test_data_path,target_variable)
    model_trainer= ModelTrainer()
    r2_score = model_trainer.initiate_model_training(train_array, test_array)
    print(r2_score)



