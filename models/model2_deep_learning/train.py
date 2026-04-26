"""
Model 2: Readmission Risk — Deep Neural Network — Train
========================================================
Trains a DNN on tabular encounter data using TensorFlow/Keras.
Saves model, scaler, imputer, and feature cols for use by predict.py.
Compares performance against Model 1 (XGBoost).

Usage (from project root):
    python models/model2_deep_learning/train.py

Outputs:
    models/model2_deep_learning/saved_model/dnn_model.keras
    models/model2_deep_learning/saved_model/scaler.joblib
    models/model2_deep_learning/saved_model/imputer.joblib
    models/model2_deep_learning/saved_model/feature_cols.joblib
    models/model2_deep_learning/saved_model/metrics.json
    models/model2_deep_learning/saved_model/training_curves.png
    models/model2_deep_learning/saved_model/confusion_matrix.png
"""

import sys
import json
import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# suppress tensorflow info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import (
    load_raw_data,
    clean_data,
    engineer_features,
    split_data,
)

SAVED_MODEL_DIR = MODEL_DIR / 'saved_model'
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = 'readmission_binary'
DROP_COLS = ['encounter_id', 'patient_nbr', 'Unnamed: 0', 'readmitted']

# update this with actual model 1 result for comparison printout
MODEL1_AUC = 0.6865


# ── data loading ───────────────────────────────────────────────────────────

def load_and_prepare():
    """Load raw data, run full pipeline, return X and y."""
    print('loading raw data...')
    df = load_raw_data('patient_encounters_2023.csv')
    print(f'  raw shape: {df.shape}')

    print('cleaning...')
    df = clean_data(df)

    print('engineering features...')
    df = engineer_features(df)

    print(f'  processed shape: {df.shape}')

    if TARGET_COL not in df.columns:
        raise ValueError(f"target column '{TARGET_COL}' not found")

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[c for c in DROP_COLS + [TARGET_COL] if c in df.columns])

    obj_cols = X.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        print(f'  one-hot encoding: {obj_cols}')
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    X = X.apply(pd.to_numeric, errors='coerce')

    print(f'  features: {X.shape[1]}')
    print(f'  class balance: {y.value_counts().to_dict()}')
    return X, y


# ── architecture ───────────────────────────────────────────────────────────

def build_dnn(n_features, learning_rate=0.001):
    """
    Build DNN architecture.
    Dense(256) -> BN -> Dropout -> Dense(128) -> BN -> Dropout -> Dense(64) -> Dropout -> Output

    Naming the AUC metric explicitly so the history key is always 'auc'
    regardless of TensorFlow version — avoids the 'auc_1' key issue.
    """
    model = Sequential([
        Input(shape=(n_features,)),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        # name AUC explicitly — this ensures history key is 'auc' not 'auc_1'
        metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy']
    )

    return model


# ── training ───────────────────────────────────────────────────────────────

def train_dnn(model, X_train, y_train, X_val, y_val, class_weight):
    """Train the DNN with early stopping and LR reduction."""

    early_stop = EarlyStopping(
        monitor='val_auc',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=256,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    return history


# ── plots ──────────────────────────────────────────────────────────────────

def plot_training_curves(history):
    """Save training AUC and loss curves."""

    # find AUC keys dynamically — handles any TF version naming
    all_keys = list(history.history.keys())
    auc_key = next((k for k in all_keys if 'auc' in k and 'val' not in k), None)
    val_auc_key = next((k for k in all_keys if 'auc' in k and 'val' in k), None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if auc_key and val_auc_key:
        axes[0].plot(history.history[auc_key], label='train AUC')
        axes[0].plot(history.history[val_auc_key], label='val AUC')
        axes[0].axhline(y=0.70, color='red', linestyle='--', label='benchmark (0.70)')
        axes[0].axhline(y=0.80, color='green', linestyle='--', label='stretch (0.80)')
    axes[0].set_title('AUC over epochs — Model 2 DNN')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('AUC')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='train loss')
    axes[1].plot(history.history['val_loss'], label='val loss')
    axes[1].set_title('loss over epochs — Model 2 DNN')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')
    axes[1].legend()

    plt.tight_layout()
    out_path = SAVED_MODEL_DIR / 'training_curves.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  training curves saved → {out_path}')


def plot_confusion_matrix(y_val, y_pred):
    """Save confusion matrix."""
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Readmit', 'Readmit'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title('Confusion Matrix — Model 2 DNN')
    plt.tight_layout()
    out_path = SAVED_MODEL_DIR / 'confusion_matrix.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  confusion matrix saved → {out_path}')


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print('=' * 55)
    print('  Model 2: Readmission Risk — Deep Neural Network')
    print('=' * 55)
    print(f'  TensorFlow : {tf.__version__}')
    print(f'  GPU        : {len(tf.config.list_physical_devices("GPU")) > 0}')
    print('=' * 55)

    # 1. load and prepare
    X, y = load_and_prepare()
    feature_names = X.columns.tolist()

    # 2. split — same random_state=42 as model 1 for fair comparison
    X_train_df, X_val_df, y_train, y_val = split_data(X, y)
    print(f'\ntrain: {len(X_train_df):,}  |  val: {len(X_val_df):,}')

    # 3. impute and scale
    # fit ONLY on training data — transform val with training stats
    print('\nscaling features...')
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train_df)
    X_val_imp = imputer.transform(X_val_df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    # 4. class weights
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: float(neg / pos)}
    print(f'class weights: {class_weight}')

    # 5. build and train
    print('\nbuilding DNN...')
    model = build_dnn(X_train_scaled.shape[1])
    model.summary()

    print('\ntraining...')
    history = train_dnn(
        model, X_train_scaled, y_train,
        X_val_scaled, y_val, class_weight
    )

    # 6. evaluate
    y_proba = model.predict(X_val_scaled, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)
    dnn_auc = roc_auc_score(y_val, y_proba)

    diff = dnn_auc - MODEL1_AUC
    print(f"\n{'='*55}")
    print(f'  Model 2 DNN AUC-ROC : {dnn_auc:.4f}  {"✅" if dnn_auc >= 0.70 else "❌ below 0.70"}')
    print(f'  Model 1 XGB AUC-ROC : {MODEL1_AUC:.4f}')
    print(f'  Difference          : {diff:+.4f} ({"DNN better" if diff > 0 else "XGBoost better"})')
    print(f'  Benchmark           : {"✅ met (>0.70)" if dnn_auc >= 0.70 else "❌ below 0.70"}')
    print(f'  Stretch goal        : {"🎯 met (>0.80)!" if dnn_auc >= 0.80 else "not yet (>0.80)"}')
    print('=' * 55)

    print(f'\n{classification_report(y_val, y_pred, target_names=["No Readmit", "Readmit"])}')

    # 7. save all artifacts
    # predict.py needs all four of these to work on new data
    model.save(SAVED_MODEL_DIR / 'dnn_model.keras')
    joblib.dump(scaler, SAVED_MODEL_DIR / 'scaler.joblib')
    joblib.dump(imputer, SAVED_MODEL_DIR / 'imputer.joblib')
    joblib.dump(feature_names, SAVED_MODEL_DIR / 'feature_cols.joblib')
    print(f'\nmodel saved → {SAVED_MODEL_DIR / "dnn_model.keras"}')

    # 8. save metrics
    metrics_out = {
        'model': 'DNN',
        'auc_roc': round(dnn_auc, 4),
        'benchmark_met': dnn_auc >= 0.70,
        'stretch_goal_met': dnn_auc >= 0.80,
        'model1_auc_for_comparison': MODEL1_AUC,
        'dnn_vs_xgboost_diff': round(diff, 4),
        'epochs_trained': len(history.history['loss']),
    }
    with open(SAVED_MODEL_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)

    # 9. visualizations
    plot_training_curves(history)
    plot_confusion_matrix(y_val, y_pred)

    print('\n✅ model 2 training complete.')


if __name__ == '__main__':
    main()
