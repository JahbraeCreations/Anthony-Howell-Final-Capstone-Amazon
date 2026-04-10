"""
Shared Data Pipeline
====================
Shared data loading and preprocessing functions used across all models.
Put your common data cleaning, feature engineering, and splitting logic here.

Usage from any model:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipelines.data_pipeline import load_raw_data, preprocess, split_data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_data(filename):
    """Load a raw CSV file from data/raw/.

    Args:
        filename: Name of the CSV file (e.g., "patient_encounters_2023.csv")

    Returns:
        pandas DataFrame

    Example:
        df = load_raw_data("patient_encounters_2023.csv")
    """
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Make sure you've downloaded the data to data/raw/"
        )
    # TODO: Load and return the CSV
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    """Apply common data cleaning steps.

    Things to handle:
    - Missing value encoding (e.g., '?' -> NaN)
    - Data type conversions
    - Remove duplicates
    - Drop irrelevant columns

    Returns:
        Cleaned DataFrame
    """
    # TODO: Add your cleaning logic

    df = df.copy()

    # 1. Replace '?' with NaN across the dataframe
    df = df.replace('?', np.nan)

    # 2. Convert age brackets to numeric midpoint
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25,
        '[30-40)': 35, '[40-50)': 45, '[50-60)': 55,
        '[60-70)': 65, '[70-80)': 75, '[80-90)': 85,
        '[90-100)': 95
    }
    if 'age' in df.columns:
        df['age_numeric'] = df['age'].map(age_map)

    # 3. Exclude death/hospice discharges BEFORE target creation
    exclude_dispositions = [11, 13, 14, 19, 20, 21]
    if 'discharge_disposition_id' in df.columns:
        df = df[
            ~df['discharge_disposition_id'].isin(exclude_dispositions)
        ].copy()

    # 4. Create diagnosis category columns for later use
    diag_cols = ['diag_1', 'diag_2', 'diag_3']

    for col in diag_cols:
        if col in df.columns:
            code_str = df[col].astype(str).str.strip()

            code_num = pd.to_numeric(
                code_str.str.extract(r'^(\d{3})')[0],
                errors='coerce'
            )

            df[f'{col}_category'] = np.select(
                [
                    df[col].isna(),
                    code_str.str.startswith('250'),
                    code_num.between(390, 459),
                    code_num.between(460, 519),
                    code_num.between(520, 579),
                    code_str.str.startswith(('V', 'E'))
                ],
                [
                    'missing',
                    'diabetes',
                    'circulatory',
                    'respiratory',
                    'digestive',
                    'external'
                ],
                default='other'
            )



    return df



def engineer_features(df):
    """Create new features from existing columns.
    Assumes clean_data() has already run first.

    Examples:
    - Parse datetime columns -> hour, day_of_week, month
    - Create binary flags from categorical data
    - Bin continuous variables into categories
    - Interaction features

    Returns:
        DataFrame with new feature columns
    """
    # TODO: Add your feature engineering

    df = df.copy()

    # 1. Create binary readmission target
    if 'readmitted' in df.columns:
        df['readmission_binary'] = (df['readmitted'] != 'NO').astype(int)

    # 2. Medication-derived features MUST happen before medication encoding
    med_cols_all = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    med_cols_all = [c for c in med_cols_all if c in df.columns]

    if med_cols_all:
        # Count medications that are active (anything not "No")
        df['num_active_meds'] = (df[med_cols_all] != 'No').sum(axis=1)

        # Count medications changed (same logic as your original intent)
        df['n_meds_changed'] = df[med_cols_all].apply(
            lambda row: sum(1 for v in row if v != 'No'),
            axis=1
        )

        # Binary flag for any medication change
        df['any_med_changed'] = (df['n_meds_changed'] > 0).astype(int)

        # Count medications increased
        df['n_meds_increased'] = df[med_cols_all].apply(
            lambda row: sum(1 for v in row if v == 'Up'),
            axis=1
        )

    # 3. Clinical complexity score
    complexity_parts = {
        'num_lab_procedures': 50,
        'num_procedures': 6,
        'num_medications': 20,
        'number_diagnoses': 9,
        'number_emergency': 3,
        'number_inpatient': 5
    }

    if all(col in df.columns for col in complexity_parts):
        df['complexity_score'] = sum(
            df[col].fillna(0) / divisor
            for col, divisor in complexity_parts.items()
        )

    # 4. Race: drop missing and Other, then one-hot encode
    if 'race' in df.columns:
        df = df.dropna(subset=['race']).copy()
        df = df[df['race'] != 'Other'].copy()
        df = pd.get_dummies(df, columns=['race'])

    # 5. Gender: drop unknown/invalid, map Male/Female
    if 'gender' in df.columns:
        df = df[df['gender'] != 'Unknown/Invalid'].copy()
        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

    # 6. Replace age with cleaned numeric age
    if 'age_numeric' in df.columns:
        df['age'] = df['age_numeric']
        df = df.drop(columns=['age_numeric'], errors='ignore')

    # 7. Weight: binary flag for whether weight was recorded
    if 'weight' in df.columns:
        df['weight'] = (
            df['weight'].notna() & (df['weight'] != '')
        ).astype(int)
        df = df.rename(columns={'weight': 'weight_checked'})

    # 8. Drop columns you decided not to use
    df = df.drop(columns=['payer_code'], errors='ignore')
    df = df.drop(columns=['medical_specialty'], errors='ignore')

    # 9. Use diagnosis category columns created in clean_data()
    diag_pairs = [
        ('diag_1', 'diag_1_category'),
        ('diag_2', 'diag_2_category'),
        ('diag_3', 'diag_3_category')
    ]

    for raw_col, cat_col in diag_pairs:
        if cat_col in df.columns:
            df[raw_col] = df[cat_col]

    # Remove rows where any diagnosis category is missing
    required_diag_cols = ['diag_1', 'diag_2', 'diag_3']
    if all(col in df.columns for col in required_diag_cols):
        df = df[
            ~(
                (df['diag_1'] == 'missing') |
                (df['diag_2'] == 'missing') |
                (df['diag_3'] == 'missing')
            )
        ].copy()

        # Drop helper category columns after replacing raw diagnosis cols
        df = df.drop(
            columns=['diag_1_category', 'diag_2_category', 'diag_3_category'],
            errors='ignore'
        )

        # One-hot encode diagnosis categories
        df = pd.get_dummies(
            df,
            columns=['diag_1', 'diag_2', 'diag_3'],
            drop_first=True
        )

    # 10. Encode medication status AFTER medication-derived counts are created
    med_status_map = {
        'No': 0,
        'Steady': 1,
        'Up': 2,
        'Down': 2
    }

    for col in med_cols_all:
        if col in df.columns:
            df[col] = df[col].map(med_status_map)

    # 11. Drop near-constant medication columns after any counts/encoding
    df = df.drop(
        columns=['examide', 'citoglipton', 'troglitazone'],
        errors='ignore'
    )

    # 12. Binary encode diabetesMed
    if 'diabetesMed' in df.columns:
        df['diabetesMed'] = df['diabetesMed'].map({'Yes': 1, 'No': 0})

    # 13. Binary encode change
    if 'change' in df.columns:
        df['change'] = df['change'].map({'Ch': 1, 'No': 0})

    #drop readmitted since it could cause data leakage
    df = df.drop(columns=['readmitted'], errors='ignore')

    return df


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and validation sets.

    IMPORTANT: Use stratify=y for imbalanced classification tasks.

    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion for validation (default 0.2)
        random_state: For reproducibility

    Returns:
        X_train, X_val, y_train, y_val
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def save_processed_data(df, filename):
    """Save processed data to data/processed/.

    Args:
        df: Processed DataFrame
        filename: Output filename (e.g., "encounters_processed.csv")
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


def load_processed_data(filename):
    """Load previously processed data from data/processed/.

    Args:
        filename: Name of the processed CSV file

    Returns:
        pandas DataFrame
    """
    filepath = PROCESSED_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found: {filepath}\n"
            f"Run the data pipeline first to generate processed data."
        )
    return pd.read_csv(filepath)
