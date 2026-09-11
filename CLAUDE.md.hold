# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **end-to-end machine learning system** for predicting high school graduation risk (90.6% accuracy). The project uses XGBoost for predictions, Streamlit for the web interface, and Evidently AI for model/data quality monitoring.

**Key Context:**
- Target variable: `hs_diploma` (binary: 0=at-risk, 1=diploma)
- Dataset: ~52,000 student records with 15 predictive features
- Main apps: `app_pred.py` (predictions) and `app.py` (quality reports)
- Package manager: **UV** (ultra-fast, preferred over pip)

## Essential Commands

### Development Setup
```bash
# Install dependencies (use UV, not pip)
make install-dev              # Install with dev dependencies
uv pip install -e ".[dev,test]"  # Alternative UV command

# Create virtual environment
uv venv                       # Creates .venv/
```

### Running Applications
```bash
# Prediction web app (port 8501)
make run-app
uv run streamlit run app_pred.py

# Data/model quality reports viewer
make run-reports
uv run streamlit run app.py

# Run with auto-reload in development
make dev
```

### Training & Testing
```bash
# Train model (runs full pipeline: ingestion → transformation → training)
make train
uv run python src/components/data_ingestion_eda.py

# Run test suite
make test
uv run pytest tests/
```

### Code Quality
```bash
# Format code (Black + Ruff)
make format
uv run black src/
uv run ruff check --fix src/

# Run linters
make lint
uv run ruff check src/
uv run black --check src/
uv run mypy src/

# Clean build artifacts
make clean
```

### Pre-commit Hooks
```bash
make setup-hooks              # Install hooks
make pre-commit               # Run on all files
```

## Architecture

### ML Pipeline Flow

The system follows a **three-stage modular pipeline**:

```
1. Data Ingestion (data_ingestion_eda.py)
   - Loads data/raw/data_2009.csv
   - Splits into train/test (80/20)
   - Generates Evidently quality reports
   - Saves to data/raw/train/ and data/raw/test/

2. Data Transformation (data_transformation.py)
   - Preprocessing pipeline with ColumnTransformer
   - Numerical features: KNNImputer → StandardScaler
   - Categorical features: SimpleImputer (most_frequent) → OneHotEncoder
   - Saves fitted preprocessor to artifacts/preprocessor.pkl

3. Model Training (model_trainer.py)
   - Trains multiple models (configurable in config/params.yaml)
   - Uses Factory Pattern for safe model instantiation (NO eval())
   - Hyperparameter tuning with hyperopt (Bayesian optimization)
   - Saves best model to artifacts/model.pkl
   - Generates performance reports
```

**Important:** Each stage depends on the previous one completing successfully. Artifacts are stored in `artifacts/` and must exist for predictions to work.

### Prediction Pipeline

Located in `src/pipeline/predict_pipeline.py`:

```python
PredictPipeline.predict(data: pd.DataFrame)
  → Loads artifacts/model.pkl and artifacts/preprocessor.pkl
  → Validates input (checks for empty/invalid data)
  → Transforms features using preprocessor
  → Returns prediction + probability
```

**Input requirements:**
- Must be pandas DataFrame
- Must contain all 15 features (see Features section below)
- Categorical values must match training data format

### Configuration System

All configuration is centralized in `config/params.yaml`:

```yaml
base:
  random_state: 67           # Reproducibility seed
  target_col: hs_diploma     # Target variable name

models:                      # Enable/disable models for training
  logistic_regression: true
  svc: true
  decision_tree: true
  random_forest: true
  naive_bayes: true
  knn: true
  xgboost: true              # Best performer (90.6% accuracy)

model_tuning:                # Hyperparameter grids
  xg_boost:
    cv: 5
    param_grid: {...}
```

**Loading config:** Use `src.config_loader.get_config(config_path)` singleton pattern.

### Directory Structure

```
artifacts/          # Trained models and preprocessors (generated)
├── model.pkl       # Best model (XGBoost)
├── preprocessor.pkl  # Fitted preprocessing pipeline
└── *.html          # Evidently quality reports

config/
└── params.yaml     # Central configuration (models, hyperparameters)

data/
├── raw/            # Original data and splits
│   ├── data_2009.csv
│   ├── train/
│   └── test/
└── processed/      # (future: transformed datasets)

src/
├── components/     # ML pipeline stages
│   ├── data_ingestion_eda.py      # Stage 1: Load & split
│   ├── data_transformation.py     # Stage 2: Preprocessing
│   └── model_trainer.py           # Stage 3: Training
├── pipeline/
│   └── predict_pipeline.py        # Inference API
├── config_loader.py  # YAML configuration loader
├── utils.py          # save_object, load_object, evaluate_models
├── exception.py      # CustomException wrapper
└── logger.py         # Logging configuration

app_pred.py         # Streamlit prediction interface
app.py              # Streamlit reports viewer
```

## Key Features

### 15 Predictive Features

**Demographics (3):**
- `male` (binary: yes/no)
- `race_ethnicity` (categorical: White, Black, Hispanic, Asian, Other)
- `frpl` (binary: yes/no - Free/Reduced Price Lunch)

**Support Services (3):**
- `iep` (binary: yes/no - Individualized Education Program)
- `ell` (binary: yes/no - English Language Learner)
- `ever_alternative` (binary: yes/no - Alternative school history)

**Academic Performance (4):**
- `gpa` (continuous: 0.0-4.0)
- `math_ss` (continuous: standardized math score)
- `read_ss` (continuous: standardized reading score)
- `ap_ever_take_class` (binary: yes/no)

**Attendance (1):**
- `pct_days_absent` (continuous: percentage)

**ACT Scores (4):**
- `scale_score_11_eng` (continuous: 11th grade ACT English)
- `scale_score_11_math` (continuous: 11th grade ACT Math)
- `scale_score_11_read` (continuous: 11th grade ACT Reading)
- `scale_score_11_comp` (continuous: 11th grade ACT Composite)

## Important Implementation Details

### Model Factory Pattern

**DO NOT use `eval()` for model instantiation.** The codebase uses a safe factory pattern in `model_trainer.py`:

```python
# WRONG (security risk):
model = eval(f"{model_name}()")

# CORRECT (factory pattern):
MODEL_FACTORY = {
    'LogisticRegression': LogisticRegression,
    'SVC': SVC,
    # ...
}
model = MODEL_FACTORY[model_name]()
```

### Artifact Serialization

Models and preprocessors are saved using `dill` (not pickle) for better compatibility:

```python
from src.utils import save_object, load_object

# Save
save_object(file_path="artifacts/model.pkl", obj=model)

# Load
model = load_object(file_path="artifacts/model.pkl")
```

### Logging & Exception Handling

All components use centralized logging and exception handling:

```python
from src.logger import logging
from src.exception import CustomException

try:
    logging.info("Starting operation...")
    # ... code ...
except Exception as e:
    raise CustomException(e, sys)
```

### Evidently AI Integration

Data and model quality monitoring is built-in:

- **Reports:** Generated during training and saved to `artifacts/*.html`
- **Metrics:** Data drift, target drift, data quality, model performance
- **Tests:** Automated test suites for data stability
- **Column Mapping:** Define in `data_ingestion_eda.py` for proper feature typing

## Development Guidelines

### When Modifying the Pipeline

1. **Data Ingestion Changes:**
   - Update train/test split logic in `data_ingestion_eda.py`
   - Regenerate artifacts: `make train`
   - Check Evidently reports in `artifacts/`

2. **Feature Engineering:**
   - Modify preprocessing in `data_transformation.py`
   - Update both numerical and categorical pipelines
   - Test with: `pytest tests/` (when tests exist)

3. **Model Changes:**
   - Enable/disable models in `config/params.yaml`
   - Add new models to MODEL_FACTORY in `model_trainer.py`
   - Update hyperparameter grids in `config/params.yaml`

4. **Prediction API:**
   - Modify `predict_pipeline.py`
   - Ensure input validation is comprehensive
   - Test with sample data from `features` dict in `data_ingestion_eda.py`

### Code Style

- **Line length:** 100 characters (Black + Ruff)
- **Type hints:** Use throughout (checked with MyPy)
- **Docstrings:** Required for all public functions
- **Imports:** Auto-sorted by Ruff (replaces isort)

### Testing

Tests should cover:
- Input validation in prediction pipeline
- Preprocessing transformations
- Model loading/saving
- Configuration loading

Run with: `make test` or `uv run pytest tests/ -v --cov=src`

## Common Workflows

### Adding a New Model

1. Add model import to `model_trainer.py`
2. Add to `MODEL_FACTORY` dictionary
3. Enable in `config/params.yaml` under `models:`
4. (Optional) Add hyperparameter grid under `model_tuning:`
5. Retrain: `make train`

### Updating Configuration

Edit `config/params.yaml` → Changes take effect on next training run (no code changes needed)

### Debugging Predictions

```python
from src.pipeline.predict_pipeline import PredictPipeline
import pandas as pd

# Use sample data from data_ingestion_eda.py
features = {
    'male': 'yes',
    'race_ethnicity': 'white',
    # ... (15 features total)
}
data = pd.DataFrame([features])

pipeline = PredictPipeline()
prediction, probability = pipeline.predict(data)
print(f"Prediction: {prediction}, Probability: {probability}")
```

### Regenerating Reports

Reports are automatically generated during training, but can be regenerated:
1. Run: `make train` (full pipeline)
2. Or directly: `uv run python src/components/data_ingestion_eda.py`
3. View in browser: `make run-reports`

## Package Management

**IMPORTANT:** This project uses **UV** (not pip) for package management:

- **10-100x faster** than pip
- **Deterministic** dependency resolution
- **Better error messages**

Common UV commands:
```bash
uv pip install <package>       # Install package
uv pip install -e .             # Install project in editable mode
uv pip list                     # List installed packages
uv pip compile pyproject.toml   # Generate requirements.txt
uv venv                         # Create virtual environment
```

**Fallback to pip:** If UV is unavailable, standard pip commands work, but are slower.

## Performance Notes

- **Training time:** ~5-10 minutes on typical hardware (42K training samples)
- **Prediction latency:** <100ms per student
- **Memory usage:** ~500MB for model + preprocessor
- **Best model:** XGBoost (90.6% accuracy, 94.7% precision, 93.9% recall)

## Data Source

Synthetic educational data from **Strategic Data Project** (Harvard CEPR) for analytics research. Data is anonymized and does not contain real student information.
