"""
Model 2: Readmission Risk — Deep Neural Network — Predict
==========================================================
Loads trained DNN and runs predictions on raw unseen encounter data.
Applies full preprocessing pipeline automatically — no manual steps needed.

Usage (from project root):
    python models/model2_deep_learning/predict.py

Input:
    test_data/*.csv  (raw encounter data, same schema as training data)

Output:
    test_data/model2_results.csv

Output columns (matches output_templates/model2_results_template.csv):
    id | prediction | probability | confidence
"""

import sys
import os
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import (
    clean_data,
    engineer_features,
    find_test_csv,
)

SAVED_MODEL_DIR = MODEL_DIR / 'saved_model'
TEST_DATA_DIR = PROJECT_ROOT / 'test_data'
OUTPUT_PATH = TEST_DATA_DIR / 'model2_results.csv'


# ── confidence helper ──────────────────────────────────────────────────────

def probability_to_confidence(proba):
    """
    Convert probability to confidence score (0-1).
    confidence = how far the probability is from 0.5 (decision boundary)
    prob=0.95 -> confidence=0.90  (very confident)
    prob=0.50 -> confidence=0.00  (completely uncertain)
    """
    return np.maximum(proba, 1 - proba)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print('=' * 55)
    print('  Model 2: Readmission Risk — DNN Inference')
    print('=' * 55)

    # 1. check artifacts exist
    model_path = SAVED_MODEL_DIR / 'dnn_model.keras'
    if not model_path.exists():
        print(f'❌ model not found at {model_path}')
        print('   run train.py first.')
        sys.exit(1)

    # 2. load model and all preprocessing artifacts
    print('loading model and artifacts...')
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(SAVED_MODEL_DIR / 'scaler.joblib')
    imputer = joblib.load(SAVED_MODEL_DIR / 'imputer.joblib')
    feature_cols = joblib.load(SAVED_MODEL_DIR / 'feature_cols.joblib')
    print(f'  model loaded | expecting {len(feature_cols)} features')

    # 3. find test data
    print('\nlocating test data...')
    test_csv = find_test_csv(
        TEST_DATA_DIR,
        expected_columns=['encounter_id', 'patient_nbr'],
        name_hint='encounter'
    )
    print(f'  found: {test_csv.name}')

    # 4. load raw data
    raw_df = pd.read_csv(test_csv, low_memory=False)
    print(f'  rows loaded: {raw_df.shape[0]:,}')

    # 5. preprocess — same pipeline as training
    print('\npreprocessing...')
    df = clean_data(raw_df.copy())
    df = engineer_features(df)

    # get encounter IDs after filtering
    encounter_ids = df['encounter_id'].values if 'encounter_id' in df.columns else np.arange(len(df))

    # drop target and ID columns
    drop_cols = ['readmission_binary', 'readmitted', 'encounter_id', 'patient_nbr', 'Unnamed: 0']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # one-hot encode remaining object columns
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    df = df.apply(pd.to_numeric, errors='coerce')

    # align to training feature set
    # add missing columns as 0, drop extras
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    print(f'  rows after preprocessing: {len(df):,}')

    # 6. apply saved imputer and scaler
    # MUST use saved artifacts — not fit new ones on test data
    X = imputer.transform(df)
    X = scaler.transform(X)

    # 7. predict
    print('running predictions...')
    probabilities = model.predict(X, verbose=0).flatten()
    predictions = (probabilities >= 0.5).astype(int)
    confidence = probability_to_confidence(probabilities)

    # 8. build output — must match output_templates/model2_results_template.csv
    results = pd.DataFrame({
        'id': encounter_ids[:len(predictions)],
        'prediction': predictions,
        'probability': np.round(probabilities, 4),
        'confidence': np.round(confidence, 4),
    })

    # 9. save
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f'\n✅ predictions saved → {OUTPUT_PATH}')
    print(f'   total predictions  : {len(results):,}')
    print(f'   predicted readmit  : {predictions.sum():,} ({predictions.mean()*100:.1f}%)')
    print(f'   avg confidence     : {confidence.mean():.3f}')


if __name__ == '__main__':
    main()
