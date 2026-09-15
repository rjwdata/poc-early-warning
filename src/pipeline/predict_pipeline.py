# needs to be customized for each model type
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, engineer_features
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def validate_input(self, data):
        """
        Validates input data before making predictions.

        Args:
            data: Input data to validate

        Raises:
            ValueError: If input data is invalid
        """
        # Check if data is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Input must be a pandas DataFrame, got {type(data).__name__}")

        # Check if DataFrame is empty
        if data.empty:
            raise ValueError("Input DataFrame is empty")

        # Check if DataFrame has any rows
        if len(data) == 0:
            raise ValueError("Input DataFrame has no rows")

        # Check for all NaN rows
        if data.isnull().all(axis=1).any():
            raise ValueError("Input contains rows with all NaN values")

        return True

    def predict(self,data):
        try:
            # Validate input data
            self.validate_input(data)

            logging.info('Engineering derived features')
            data = engineer_features(data)

            logging.info('Loading model and preprocessor')
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            logging.info("Loading model and preprocessor artifacts")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            logging.info("Model and preprocessor loaded successfully")
            data_scaled=preprocessor.transform(data)
            preds=model.predict(data_scaled)
            probs =model.predict_proba(data_scaled)[:,1]

            logging.info(f"Prediction completed successfully for {len(data)} records")
            return preds, probs

        except Exception as e:
            raise CustomException(e,sys)
