import os
import sys
from src.exception import CustomException
from  src.logger import logging
import pandas as pd

import argparse
import yaml

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info('Read the data set as dataframe')

            df.columns = df.columns.str.replace(' ', '_').str.replace('/', '_')
            logging.info('Removed spaces and characters from column names')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            
            logging.info('train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state = 67)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('train test split completed')

            return(
               self.ingestion_config.train_data_path,
               self.ingestion_config.test_data_path,

            )
        
        except Exception as e:
             raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_set,test_set=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_set,test_set)
    
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
