"""
Constants module for the POC Early Warning System.
Contains all hardcoded values, magic numbers, and configuration paths.
"""
import os
from pathlib import Path

# Configuration paths
CONFIG_DIR = "config"
CONFIG_FILE = "params.yaml"
CONFIG_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE)

# Artifact paths
ARTIFACTS_DIR = "artifacts"
MODEL_FILE = "model.pkl"
PREPROCESSOR_FILE = "preprocessor.pkl"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, MODEL_FILE)
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, PREPROCESSOR_FILE)

# Data paths
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
TRAIN_DATA_DIR = os.path.join(RAW_DATA_DIR, "train")
TEST_DATA_DIR = os.path.join(RAW_DATA_DIR, "test")

# Report paths
REPORTS_DIR = "reports"

# Model training constants
RANDOM_STATE = 67
TEST_SIZE = 0.2
CV_FOLDS = 10

# Feature names
TARGET_COLUMN = "hs_diploma"

NUMERIC_FEATURES = [
    "math_ss",
    "read_ss",
    "pct_days_absent",
    "gpa",
    "scale_score_11_eng",
    "scale_score_11_math",
    "scale_score_11_read",
    "scale_score_11_comp",
    "attendance_rate"  # derived in src.utils.engineer_features
]

CATEGORICAL_FEATURES = [
    "male",
    "race_ethnicity",
    "frpl",
    "iep",
    "ell",
    "ever_alternative",
    "ap_ever_take_class"
]

# Model evaluation metrics
SCORING_METRIC = "accuracy"

# Logging format
LOG_FORMAT = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
