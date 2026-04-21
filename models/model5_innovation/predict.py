#!/usr/bin/env python3
"""
Model 5: Innovation — Prediction Script
=========================================
Loads your trained model and generates predictions on test data.

Usage: python predict.py
Output: test_data/model5_results.csv
"""
#to pull clean data and engineer features from data_pipeline
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared pipeline functions
from pipelines.data_pipeline import clean_data, engineer_features

# Paths
MODEL_PATH = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model5_results.csv"


def load_model():
    """Load your trained model from saved_model/.

    This is your team's innovation model — use whatever approach you chose.
    """
    import joblib
    filepath = MODEL_PATH / "model.joblib"
    return joblib.load(filepath)


def predict(model, test_df):
    """Generate predictions on test data.

    
        

    metric_name and metric_value are your custom evaluation metric.
    For example: metric_name="f1_weighted", metric_value=0.85
    """

    df = test_df.copy()

    df = clean_data(df)
    df = engineer_features(df)

    df_result = df[['encounter_id']].copy()
    df_result = df_result.rename(columns={"encounter_id": "id"}) 

    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone',
        'num_active_meds', 'n_meds_changed', 'n_meds_increased'
    ]

    existing_med_cols = [col for col in med_cols if col in df.columns]
    df = df.drop(columns=existing_med_cols, errors="ignore")

    df = df.drop(columns=["diabetesMed", "encounter_id"], errors="ignore")

    #check to delete later
    if list(df.columns) != list(model.feature_names_in_):
        raise ValueError("Test columns do not exactly match training columns.")

    confidence_scores = model.predict_proba(df)[:, 1]
    predictions = model.predict(df)

    df_result["prediction"] = predictions
    df_result["confidence"] = confidence_scores
    df_result["metric_name"] = "f1_score"
    df_result["metric_value"] = np.nan

    # TODO: Run your model on the test data
    return df_result


def main():
    # Load model
    model = load_model()

    test_df = pd.read_csv(TEST_DATA_DIR / "test_data_file.csv")

    results = predict(model, test_df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    


    # Load test data
    # TODO: Update this path to match your test data file
    # test_df = pd.read_csv(TEST_DATA_DIR / "test_data_file.csv")

    # Generate predictions
    # predictions = predict(model, test_df)

    # Save results — MUST match output template exactly
    # results = pd.DataFrame({
    #     "id": test_df["id"],
    #     "prediction": predictions,
    #     "confidence": confidence_scores,
    #     "metric_name": "your_custom_metric",
    #     "metric_value": metric_score,
    # })
    # results.to_csv(OUTPUT_FILE, index=False)

    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
