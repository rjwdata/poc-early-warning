### regression
### needs updating - question is whether we select on tuned model or type of model 
### tuning parameters may need to be updated on project

import os
import sys
import pandas as pd
from dataclasses import dataclass


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'Baseline': 0,
                'Logistic Regression': LogisticRegression(),
                'Support Vector Machines': LinearSVC(),
                'Decision Trees': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Naive Bayes': GaussianNB(),
                'K-Nearest Neighbor': KNeighborsClassifier(),
                'xgboost': xgb.XGBClassifier(objective="binary:logistic", random_state=42)
            }
            params={}

            model_report =evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            ## To get best model score from dict
            print(model_report)

            ## To get best model name from df
            max_index = model_report['Accuracy'].idxmax()
            max_value = model_report['Accuracy'].max()

            
            best_model = models[max_index]

            if max_value < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            logging.info(f'{max_index} with an accruacy of {max_value}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            #predicted=best_model.predict(X_test)

            #r2_square = r2_score(y_test, predicted)
            #feature_names = ['gender', 'race/ethnicity', 'parental_level_of_education',
            #                 'lunch','test_preparation_course', 'reading_score', 'writing_score']
            #coefs = pd.DataFrame(
            #      best_model.coef_,
            #      columns=["Coefficients"])
            #return r2_square, coefs
                
        except Exception as e:
            raise CustomException(e,sys)
