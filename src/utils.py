import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from pathlib import Path
from typing import Dict
from typing import List
from typing import Text


class EntityNotFoundError(Exception):
    """EntityNotFoundError"""

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

### This function needs to be updated on a case by case basis     
#def evaluate_models(X_train, y_train, X_test, y_test, models, param):
#    try:
#        report = {}
#
#        for i in range(len(list(models))):
#            model = list(models.values())[i]
#            para=param[list(models.keys())[i]]
#
#            gs = GridSearchCV(model,para,cv=3)
#            gs.fit(X_train,y_train)
#
#            model.set_params(**gs.best_params_)
#            model.fit(X_train,y_train)
#
#            model.fit(X_train, y_train)
#
#            y_train_pred = model.predict(X_train)
#
#            y_test_pred = model.predict(X_test)
#
#            train_model_score = r2_score(y_train, y_train_pred)
#
#            test_model_score = r2_score(y_test, y_test_pred)
#
#            report[list(models.keys())[i]] = test_model_score
#
#        return report
#
#    except Exception as e:
#        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def list_periods(reports_dir: Path) -> List[Text]:
    """List periods subdirectories inside reports directory.

    Args:
        reports_dir (Path): Reports directory path.

    Raises:
        EntityNotFoundError: If reports directory does not exist.

    Returns:
        List[Text]: List of periods subdirectories
    """

    try:
        return sorted(
            list(filter(lambda e: (reports_dir / e).is_dir(), os.listdir(reports_dir)))
        )
    except FileNotFoundError as e:
        raise EntityNotFoundError(e)


def period_dir_to_dates_range(period_dir_name: Text) -> Text:
    """_summary_

    Args:
        period_dir_name (Text): _description_

    Returns:
        Text: _description_
    """

    return period_dir_name.replace("_", " - ")


def get_report_name(path: Path) -> Text:
    """Convert report path to human readable name.

    Args:
        path (Path): Report path.

    Returns:
        Text: human readable name.
    """

    name: Text = path.with_suffix("").name.replace("_", " ").capitalize()

    return name


def get_reports_mapping(reports_dir: Text) -> Dict[Text, Path]:
    """Build dictionary where human readable names corresponds to paths.
    Note: each directory gets suffix ` (folder)`.

    Args:
        paths (List[Path]): List of paths.

    Returns:
        Dict[Text, Path]: Dictionary with structure:
        {
            <Name>: <path>
        }

    Examples:
    >>> paths = [
        'reports/2011-02-12_2011-02-18/data_quality',
        'reports/2011-02-12_2011-02-18/model_performance',
        'reports/2011-02-12_2011-02-18/data_drift.html',
        'reports/2011-02-12_2011-02-18/data_quality.html',
        'reports/2011-02-12_2011-02-18/model_performance.html',
        'reports/2011-02-12_2011-02-18/target_drift.html'
    ]
    >>> report_paths_to_names(paths)
    {
        'Data drift': 'Path(reports/2011-02-12_2011-02-18/data_drifts.html)',
        'Data quality(folder)': 'Path(reports/2011-02-12_2011-02-18/data_quality)',
        'Data quality': 'Path(reports/2011-02-12_2011-02-18/data_quality.html)',
        'Model performance (folder)': 'Path(reports/2011-02-12_2011-02-18/model_performance)',
        'Model performance': 'Path(reports/2011-02-12_2011-02-18/model_performance.html)',
        'Target drift': 'Path(reports/2011-02-12_2011-02-18/target_drift.html)'
    }
    """

    names: List[Text] = []
    paths: List[Path] = []

    for filename in os.listdir(reports_dir):
        if not filename.startswith("."):
            paths.append(Path(f"{reports_dir}/{filename}"))
    paths.sort()

    for path in paths:
        name: Text = get_report_name(path)
        if path.is_dir():
            name += " (folder)"
        names.append(name)

    return dict(zip(names, paths))
