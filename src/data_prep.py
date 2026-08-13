"""
data_prep.py — Stage 2: data cleaning, feature engineering, leakage-safe
preprocessing and splitting for the Hospital Readmission pipeline.

Implement the functions below; they are imported by Data_Preparation.ipynb and by
src/train.py. Expected end state: a model frame of ~69,973 rows at ~9% positive, and a
stratified train/val/test split with the preprocessor fit on the TRAIN split only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import config as cfg

ID_CODE_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]


def load_raw():
    """
    Load the raw hospital readmission dataset.

    '?' values are interpreted as missing values (NaN).
    """

    return pd.read_csv(
        cfg.RAW_CSV,
        na_values=["?"]
    )


def clean(df):
    """
    Stage 2.1 — Data Cleaning

    1. Convert '?' to NaN for robustness.
    2. Keep only the first encounter for each patient.
    3. Remove expired/hospice discharge records.
    4. Create the binary 30-day readmission target.
    5. Drop identifiers, unusable columns and the original target column.
    """

    data = df.copy()

    data = (
        data.sort_values("encounter_id")
            .drop_duplicates(
                subset="patient_nbr",
                keep="first"
            )
            .copy()
    )

    data = data[
        ~data["discharge_disposition_id"].isin(
            cfg.EXPIRED_HOSPICE_DISPOSITIONS
        )
    ].copy()


    data[cfg.TARGET] = (
        data["readmitted"] == "<30"
    ).astype(int)


    drop_cols = [
        col for col in cfg.DROP_COLS
        if col in data.columns
    ]

    data = data.drop(
        columns=drop_cols,
        errors="ignore"
    )


    data = data.drop(
        columns=["readmitted"],
        errors="ignore"
    )


    return data.reset_index(drop=True)


def engineer_features(df):
    """
    Stage 2.2 — Feature Engineering

    2.2.1:
        Group ICD-9 diagnosis codes into broad clinical categories.

    2.2.2:
        Create:
        - numeric age midpoint
        - service_utilization
        - num_med_changes
        - grouped medical_specialty
    """

    data = df.copy()

    
    # 2.2.1 — ICD-9 diagnosis grouping
    

    def group_icd9(code):
        """
        Convert ICD-9 diagnosis codes into broad clinical groups.

        Categories required for the capstone:
        - Diabetes
        - Circulatory
        - Respiratory
        - Injury
        - Other
        - Missing
        """

        if pd.isna(code):
            return "Missing"

        code = str(code).strip()

        # V-codes / E-codes cannot be directly interpreted
        # as the standard numeric ICD-9 ranges used below.
        if code.startswith(("V", "E")):
            return "Other"

        try:
            code_num = float(code)
        except ValueError:
            return "Other"

        # Diabetes mellitus
        if 250 <= code_num < 251:
            return "Diabetes"

        # Diseases of circulatory system
        elif 390 <= code_num <= 459:
            return "Circulatory"

        # Additional circulatory symptom code
        elif 785 <= code_num < 786:
            return "Circulatory"

        # Diseases of respiratory system
        elif 460 <= code_num <= 519:
            return "Respiratory"

        # Additional respiratory symptom code
        elif 786 <= code_num < 787:
            return "Respiratory"

        # Injury and poisoning
        elif 800 <= code_num <= 999:
            return "Injury"

        else:
            return "Other"


    # Replace high-cardinality diagnosis codes with
    # clinically interpretable categories.
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in data.columns:
            data[col] = data[col].apply(group_icd9)


    
    # 2.2.2 — Additional engineered features
    # 1. Age midpoint
    # Example: [60-70) -> 65
    

    age_bounds = data["age"].astype(str).str.extract(
        r"\[(\d+)-(\d+)\)"
    )

    data["age"] = (
        age_bounds[0].astype(float)
        + age_bounds[1].astype(float)
    ) / 2


    
    # 2. Service utilisation
    # Total previous healthcare utilisation
    

    data["service_utilization"] = (
        data["number_outpatient"]
        + data["number_emergency"]
        + data["number_inpatient"]
    )


    
    # 3. Number of medication changes
    

    available_med_cols = [
        col for col in cfg.MED_COLS
        if col in data.columns
    ]

    data["num_med_changes"] = (
        data[available_med_cols]
        .isin(["Up", "Down"])
        .sum(axis=1)
    )


    
    # 4. Medical specialty top-k grouping
    

    if "medical_specialty" in data.columns:

        # Keep the 10 most frequent known specialties.
        top_specialties = (
            data["medical_specialty"]
            .dropna()
            .value_counts()
            .nlargest(10)
            .index
        )

        data["medical_specialty"] = data["medical_specialty"].apply(
            lambda x:
                "Missing"
                if pd.isna(x)
                else x
                if x in top_specialties
                else "Other"
        )


    return data

# this is for the cell Stage 2.3 
def get_feature_lists(df):
    """
    Identify numerical and categorical model features.

    - Excludes the target column.
    - Object/category columns are treated as categorical.
    - Admission/discharge/source ID codes are categorical even
      though they are stored as integers.
    """

    # All possible input features except the target
    feature_cols = [
        col for col in df.columns
        if col != cfg.TARGET
    ]

    # Object/category columns are categorical
    categorical_cols = (
        df[feature_cols]
        .select_dtypes(include=["object", "category"])
        .columns
        .tolist()
    )

    # These columns contain category codes, not continuous numbers
    for col in ID_CODE_COLS:
        if col in feature_cols and col not in categorical_cols:
            categorical_cols.append(col)

    # Everything else is treated as numeric
    numeric_cols = [
        col for col in feature_cols
        if col not in categorical_cols
    ]

    return numeric_cols, categorical_cols

#this is for the stage 2.3 implementatioin
def build_preprocessor(numeric, categorical):
    """
    Stage 2.3 — Leakage-safe feature transformation.

    Numeric pipeline:
        Median imputation -> StandardScaler

    Categorical pipeline:
        Constant imputation -> OneHotEncoder

    The ColumnTransformer is returned UNFITTED.
    It must be fitted on the training data only.
    """

    
    # 2.3.1 — Numeric preprocessing
    

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    
    # 2.3.1 — Categorical preprocessing
    

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Missing"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )


    
    # 2.3.2 — Single leakage-safe ColumnTransformer
    

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical
            )
        ],
        remainder="drop"
    )

    # IMPORTANT:
    # Do not fit here.
    # Fitting is performed only on X_train in Stage 2.4.
    return preprocessor

def get_model_frame():
    """
    Build the complete modelling dataframe by applying:

    raw data
        -> cleaning
        -> feature engineering

    Returns the final model-ready dataframe before splitting
    and preprocessing.
    """

    raw = load_raw()
    cleaned = clean(raw)
    engineered = engineer_features(cleaned)

    return engineered


def get_splits(df=None):
    """
    Stage 2.4 — Create leakage-safe stratified
    train / validation / test splits.

    Split sequence:
        1. Hold out TEST_SIZE from the full modelling frame.
        2. Hold out VAL_SIZE from the remaining training frame.

    All splits use:
        random_state = cfg.RANDOM_STATE
        stratify = target

    IMPORTANT:
    No preprocessing is fitted inside this function.
    Preprocessing is fitted on X_train only.
    """

    
    # Build model frame automatically when not supplied
    

    if df is None:
        df = get_model_frame()


    
    # Separate predictors and target
    

    X = df.drop(
        columns=[cfg.TARGET]
    )

    y = df[cfg.TARGET].astype(int)


    
    # 2.4.1 — First split: train/validation pool vs test
    

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=y
    )


    
    # Second split: train vs validation
    

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=cfg.VAL_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=y_train_val
    )


    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )
