"""
Model 1: Readmission Risk — Traditional ML — Predict
=====================================================
Loads trained model and runs predictions on raw unseen encounter data.
Applies full preprocessing pipeline automatically — no manual steps needed.

Usage (from project root):
    python models/model1_traditional_ml/predict.py

Input:
    test_data/*.csv  (raw encounter data, same schema as training data)

Output:
    test_data/model1_results.csv

Output columns (matches output_templates/model1_results_template.csv):
    id | prediction | probability | confidence
"""

import sys
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import (
    preprocess_encounters_for_prediction,
    load_pipeline_artifacts,
    find_test_csv,
)

SAVED_MODEL_DIR = MODEL_DIR / 'saved_model'
TEST_DATA_DIR = PROJECT_ROOT / 'test_data'
OUTPUT_PATH = TEST_DATA_DIR / 'model1_results.csv'


# ── confidence helper ──────────────────────────────────────────────────────

def probability_to_confidence(proba):
    """
    Convert raw probability to confidence score (0-1).
    Confidence = how far the probability is from the decision boundary (0.5).

    prob = 0.95 -> confidence = 0.90  (very confident readmission)
    prob = 0.50 -> confidence = 0.00  (completely uncertain)
    prob = 0.05 -> confidence = 0.90  (very confident no readmission)
    """
    return np.maximum(proba, 1 - proba)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print('=' * 55)
    print('  Model 1: Readmission Risk — Inference')
    print('=' * 55)

    # 1. load saved model and feature column list
    model_path = SAVED_MODEL_DIR / 'model.joblib'
    if not model_path.exists():
        print(f'❌ model not found at {model_path}')
        print('   run train.py first to generate the model.')
        sys.exit(1)

    print('loading model...')
    model = joblib.load(model_path)
    feature_cols = load_pipeline_artifacts(SAVED_MODEL_DIR)
    print(f'  model loaded | expecting {len(feature_cols)} features')

    # 2. find test data
    print('\nlocating test data...')
    test_csv = find_test_csv(
        TEST_DATA_DIR,
        expected_columns=['encounter_id', 'patient_nbr'],
        name_hint='encounter'
    )
    print(f'  found: {test_csv.name}')

    # 3. load raw data
    raw_df = pd.read_csv(test_csv, low_memory=False)
    print(f'  rows loaded: {raw_df.shape[0]:,}')

    # save encounter_id before preprocessing drops/filters rows
    # use it to build the output — need to match predictions to IDs
    original_ids = raw_df['encounter_id'].copy() if 'encounter_id' in raw_df.columns else None

    # 4. preprocess using the same pipeline as training
    # preprocess_encounters_for_prediction handles:
    #   - clean_data() and engineer_features()
    #   - aligns columns to exactly what the model was trained on
    #   - adds missing columns as 0, drops unknown extras
    print('\npreprocessing...')
    X, df_clean = preprocess_encounters_for_prediction(raw_df, feature_cols)
    print(f'  rows after preprocessing: {X.shape[0]:,}')

    # get encounter IDs from cleaned df (after filtering deceased etc.)
    if 'encounter_id' in df_clean.columns:
        encounter_ids = df_clean['encounter_id'].values
    elif original_ids is not None:
        encounter_ids = original_ids.iloc[:len(X)].values
    else:
        encounter_ids = np.arange(len(X))

    # 5. predict
    print('running predictions...')
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    confidence = probability_to_confidence(probabilities)

    # 6. build output — must exactly match output_templates/model1_results_template.csv
    results = pd.DataFrame({
        'id': encounter_ids,
        'prediction': predictions,
        'probability': np.round(probabilities, 4),
        'confidence': np.round(confidence, 4),
    })

    # 7. save
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f'\n✅ predictions saved → {OUTPUT_PATH}')
    print(f'   total predictions  : {len(results):,}')
    print(f'   predicted readmit  : {predictions.sum():,} ({predictions.mean()*100:.1f}%)')
    print(f'   avg confidence     : {confidence.mean():.3f}')


if __name__ == '__main__':
    main()
