### regression
### needs updating - question is whether we select on tuned model or type of model 
### tuning parameters may need to be updated on project

import os
import sys
import pandas as pd
from dataclasses import dataclass
import json

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from src.config_loader import get_config

## get parameters from config file
config_path = os.path.join("config", "params.yaml")
config = get_config(config_path)

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
            def create_model_from_config(model_name, model_config):
                """Safely creates a model instance from configuration without using eval()."""
                model_registry = {
                    'logistic_regression': lambda: LogisticRegression(),
                    'svc': lambda: LinearSVC(dual='auto'),
                    'decision_tree': lambda: DecisionTreeClassifier(),
                    'random_forest': lambda: RandomForestClassifier(),
                    'naive_bayes': lambda: GaussianNB(),
                    'knn': lambda: KNeighborsClassifier(),
                    'xgboost': lambda: xgb.XGBClassifier(objective="binary:logistic", random_state=42)
                }

                if model_name not in model_registry:
                    raise ValueError(f"Unknown model: {model_name}")

                return model_registry[model_name]()

            # Build models dictionary safely from config
            models = {'Baseline': 0}

            # Get models from config and instantiate them safely
            config_models = config.get('models', {})
            model_mapping = {
                'logistic_regression': 'Logistic Regression',
                'svc': 'Support Vector Machines',
                'decision_tree': 'Decision Trees',
                'random_forest': 'Random Forest',
                'naive_bayes': 'Naive Bayes',
                'knn': 'K-Nearest Neighbor',
                'xgboost': 'xgboost'
            }

            for config_key, display_name in model_mapping.items():
                if config_key in config_models and config_models[config_key]:
                    try:
                        models[display_name] = create_model_from_config(config_key, config_models[config_key])
                    except Exception as e:
                        logging.warning(f"Failed to create model {display_name}: {str(e)}")
                        continue


            all_models_results, best_model_stats, best_model = evaluate_models(X_train, y_train, models, X_test, y_test)
            
            ## To get best model score from dict
            pretty_all_models = json.dumps(all_models_results, indent=4)
            pretty_best_model = json.dumps(best_model_stats, indent=4)

            logging.info(f"All models results:\n{pretty_all_models}")
            logging.info(f"Best model stats:\n{pretty_best_model}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model training completed successfully")

        except Exception as e:
            raise CustomException(e,sys)