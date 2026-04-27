"""
Model 1: Readmission Risk — Traditional ML — Train
====================================================
Trains XGBoost, Random Forest, and Logistic Regression on encounter data.
Saves best model + feature column list for use by predict.py.

Usage (from project root):
    python models/model1_traditional_ml/train.py

Outputs:
    models/model1_traditional_ml/saved_model/model.joblib
    models/model1_traditional_ml/saved_model/feature_cols.joblib
    models/model1_traditional_ml/saved_model/metrics.json
    models/model1_traditional_ml/saved_model/shap.png
    models/model1_traditional_ml/saved_model/confusion_matrix.png
"""

import sys
import json
from xml.parsers.expat import model
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print('⚠️  xgboost not installed — run: pip install xgboost')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print('⚠️  shap not installed — run: pip install shap')

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
# this file lives at models/model1_traditional_ml/train.py
# so project root is 2 levels up
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import (
    load_raw_data,
    clean_data,
    engineer_features,
    split_data,
    save_pipeline_artifacts,
)

SAVED_MODEL_DIR = MODEL_DIR / 'saved_model'
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = 'readmission_binary'


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
        raise ValueError(
            f"target column '{TARGET_COL}' not found — "
            f"check that 'readmitted' column exists in raw data"
        )

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    # one-hot encode any remaining object columns
    obj_cols = X.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        print(f'  one-hot encoding: {obj_cols}')
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    # force all numeric
    X = X.apply(pd.to_numeric, errors='coerce')

    print(f'  features: {X.shape[1]}')
    print(f'  class balance: {y.value_counts().to_dict()}')
    return X, y


# ── models ─────────────────────────────────────────────────────────────────

def build_models(scale_pos_weight=1.0):
    """Return dict of model name -> sklearn Pipeline."""
    models = {
        'Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )),
        ]),
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )),
        ]),
    }
    if XGBOOST_AVAILABLE:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        models['XGBoost'] = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('smote', SMOTE(random_state=42)),
            ('clf', XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.05,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                 n_jobs=-1
            )),
        ])
    print('XGBoost + SMOTE pipeline ready')





    return models


# ── evaluation ─────────────────────────────────────────────────────────────

def evaluate(model, X_val, y_val, name):
    """Print and return evaluation metrics."""
    proba = model.predict_proba(X_val)[:, 1]
    preds = model.predict(X_val)
    auc = roc_auc_score(y_val, proba)
    report = classification_report(y_val, preds, output_dict=True)

    print(f"\n{'─'*50}")
    print(f'  {name}')
    print(f'  AUC-ROC  : {auc:.4f}  {"✅" if auc >= 0.70 else "❌ below 0.70 benchmark"}')
    print(f'  F1 (1)   : {report["1"]["f1-score"]:.4f}')
    print(f'  Precision: {report["1"]["precision"]:.4f}  Recall: {report["1"]["recall"]:.4f}')
    return {'auc_roc': round(auc, 4), 'report': report}


# ── shap ───────────────────────────────────────────────────────────────────

def plot_shap(model, X_train, feature_names):
    """Generate and save SHAP summary plot."""
    if not SHAP_AVAILABLE:
        print('skipping SHAP — install with: pip install shap')
        return

    print('\ngenerating SHAP values...')
    clf = model.named_steps['clf']

    # transform X through pipeline steps before the classifier
    X_transformed = X_train.copy()
    for step_name, step in model.steps[:-1]:
        if hasattr(step, 'transform'):
            X_transformed = step.transform(X_transformed)

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)
        # random forest returns a list — take class 1 (readmit)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        explainer = shap.LinearExplainer(clf, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_transformed,
        feature_names=feature_names,
        max_display=20, show=False
    )
    plt.title('SHAP Feature Importance — Model 1', fontsize=13)
    plt.tight_layout()
    out_path = SAVED_MODEL_DIR / 'shap.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  SHAP plot saved → {out_path}')


def plot_confusion_matrix(model, X_val, y_val, name):
    """Save confusion matrix plot."""
    preds = model.predict(X_val)
    cm = confusion_matrix(y_val, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Readmit', 'Readmit'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    out_path = SAVED_MODEL_DIR / 'confusion_matrix.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  confusion matrix saved → {out_path}')


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print('=' * 55)
    print('  Model 1: Readmission Risk — Traditional ML')
    print('=' * 55)

    # 1. load and prepare data
    X, y = load_and_prepare()
    feature_names = X.columns.tolist()

    # 2. train/val split
    X_train, X_val, y_train, y_val = split_data(X, y)
    print(f'\ntrain: {len(X_train):,}  |  val: {len(X_val):,}')

    # calculate scale_pos_weight for XGBoost class imbalance handling
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f'scale_pos_weight: {scale_pos_weight:.2f}')

    # 3. train all models and evaluate
    models = build_models(scale_pos_weight=scale_pos_weight)
    results = {}

    for name, model in models.items():
        print(f'\ntraining {name}...')
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_val, y_val, name)

    # 4. pick best by AUC-ROC
    best_name = max(results, key=lambda k: results[k]['auc_roc'])
    best_model = models[best_name]
    best_auc = results[best_name]['auc_roc']

    print(f"\n{'='*55}")
    print(f'  🏆 best model : {best_name}')
    print(f'  AUC-ROC      : {best_auc:.4f}')
    print(f'  benchmark    : {"✅ met (>0.70)" if best_auc >= 0.70 else "❌ below 0.70"}')
    print(f'  stretch goal : {"🎯 met (>0.80)!" if best_auc >= 0.80 else "not yet (>0.80)"}')
    print('=' * 55)

    # 5. save model and feature column list
    # feature_cols is critical — predict.py uses it to align test data columns
    joblib.dump(best_model, SAVED_MODEL_DIR / 'model.joblib')
    joblib.dump(feature_names, SAVED_MODEL_DIR / 'feature_cols.joblib')
    print(f'Feature cols saved → {SAVED_MODEL_DIR / "feature_cols.joblib"}')

    # 6. save metrics
    metrics_out = {
        'best_model': best_name,
        'auc_roc': best_auc,
        'benchmark_met': best_auc >= 0.70,
        'stretch_goal_met': best_auc >= 0.80,
        'all_models': {k: v['auc_roc'] for k, v in results.items()},
    }
    with open(SAVED_MODEL_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)

    # 7. visualizations
    plot_shap(best_model, X_train, feature_names)
    plot_confusion_matrix(best_model, X_val, y_val, best_name)

    print('\n✅ model 1 training complete.')


if __name__ == '__main__':
    main()
