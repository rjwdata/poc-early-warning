# needs to be customized for each model type
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,data):
        try:
            print('loading model and preprocessor')
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(data)
            preds=model.predict(data_scaled)
            probs =model.predict_proba(data_scaled)

            return preds, probs
        
        except Exception as e:
            raise CustomException(e,sys)
