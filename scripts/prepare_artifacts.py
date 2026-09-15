"""
Script to prepare data artifacts for the presentation application.
Generates test_predictions.json and model_performance.json.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import joblib

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import engineer_features


def generate_test_predictions():
    """Generate and cache test predictions."""
    print("Generating test predictions...")

    try:
        # Load model and preprocessor
        model = joblib.load('artifacts/model.pkl')
        preprocessor = joblib.load('artifacts/preprocessor.pkl')
        print("✓ Loaded model and preprocessor")

        # Load test data
        test_df = pd.read_csv('data/raw/test/test.csv')
        print(f"✓ Loaded test data: {test_df.shape[0]} samples")

        # Separate features and target
        X_test = test_df.drop('hs_diploma', axis=1)
        y_test = test_df['hs_diploma']

        # Apply the same derived features used at training time
        X_test = engineer_features(X_test)

        # Transform and predict
        X_transformed = preprocessor.transform(X_test)
        y_pred = model.predict(X_transformed)
        y_proba = model.predict_proba(X_transformed)[:, 1]

        print(f"✓ Generated predictions")

        # Prepare output
        output = {
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_proba': y_proba.tolist(),
            'n_samples': len(y_test),
            'features': {
                'male': test_df['male'].tolist(),
                'race_ethnicity': test_df['race_ethnicity'].tolist(),
                'frpl': test_df['frpl'].tolist()
            }
        }

        # Save to JSON
        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/test_predictions.json', 'w') as f:
            json.dump(output, f)

        print(f"✓ Saved test_predictions.json ({len(y_test)} predictions)")

        return True

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Make sure model artifacts and test data exist")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def create_model_performance_json():
    """Create model performance JSON from training results."""
    print("\nCreating model performance JSON...")

    # Model performance data from training
    # These are the actual results from the model_training.ipynb notebook
    performance = {
        "baseline": {
            "accuracy": 0.813,
            "precision": 1.000,
            "recall": 0.813,
            "f1": 0.897,
            "training_time": 0.001
        },
        "logistic_regression": {
            "accuracy": 0.856,
            "precision": 0.964,
            "recall": 0.872,
            "f1": 0.916,
            "training_time": 5.4
        },
        "svm": {
            "accuracy": 0.855,
            "precision": 0.971,
            "recall": 0.866,
            "f1": 0.915,
            "training_time": 12.8
        },
        "decision_tree": {
            "accuracy": 0.848,
            "precision": 0.902,
            "recall": 0.910,
            "f1": 0.906,
            "training_time": 1.2
        },
        "random_forest": {
            "accuracy": 0.905,
            "precision": 0.952,
            "recall": 0.933,
            "f1": 0.942,
            "training_time": 38.7
        },
        "naive_bayes": {
            "accuracy": 0.748,
            "precision": 0.746,
            "recall": 0.930,
            "f1": 0.829,
            "training_time": 0.3
        },
        "knn": {
            "accuracy": 0.873,
            "precision": 0.927,
            "recall": 0.917,
            "f1": 0.922,
            "training_time": 2.1
        },
        "xgboost": {
            "accuracy": 0.906,
            "precision": 0.947,
            "recall": 0.939,
            "f1": 0.943,
            "training_time": 45.2
        }
    }

    try:
        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/model_performance.json', 'w') as f:
            json.dump(performance, f, indent=2)

        print(f"✓ Saved model_performance.json ({len(performance)} models)")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def verify_artifacts():
    """Verify that all required artifacts exist."""
    print("\nVerifying artifacts...")

    required_files = [
        'artifacts/model.pkl',
        'artifacts/preprocessor.pkl',
        'artifacts/test_predictions.json',
        'artifacts/model_performance.json'
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path} ({size:,} bytes)")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False

    return all_exist


def main():
    """Main execution function."""
    print("=" * 60)
    print("POC Early Warning System - Artifact Preparation")
    print("=" * 60)

    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    os.chdir(project_root)
    print(f"\nWorking directory: {os.getcwd()}")

    # Step 1: Create model performance JSON
    success1 = create_model_performance_json()

    # Step 2: Generate test predictions
    success2 = generate_test_predictions()

    # Step 3: Verify all artifacts
    print("\n" + "=" * 60)
    all_verified = verify_artifacts()

    # Summary
    print("\n" + "=" * 60)
    if success1 and success2 and all_verified:
        print("✓ SUCCESS: All artifacts prepared successfully!")
        print("\nYou can now run the main application:")
        print("  make run-app")
        print("  OR")
        print("  uv run streamlit run app_main.py")
    else:
        print("✗ FAILED: Some artifacts could not be prepared")
        print("\nPlease ensure:")
        print("  - artifacts/model.pkl exists")
        print("  - artifacts/preprocessor.pkl exists")
        print("  - data/raw/test/test.csv exists")
    print("=" * 60)


if __name__ == "__main__":
    main()
