"""Data Preprocessing Module for Customer Churn Engine.

Constructs production-grade, leakage-free Scikit-Learn preprocessing pipelines
using ColumnTransformer, SimpleImputer, OneHotEncoder, and StandardScaler.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

from src.config import settings
from src.data_loader import load_raw_data


def clean_raw_dataframe(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Perform preliminary data cleaning on the raw DataFrame.

    - Strips whitespace from column names.
    - Coerces 'TotalCharges' whitespace strings to numeric NaN values.
    - Separates the target column 'Churn' into a binary Series (1=Yes, 0=No).
    - Drops 'customerID' from features to prevent spurious identifier correlation.

    Args:
        df_raw: Raw pandas DataFrame.

    Returns:
        Tuple of (X: feature DataFrame, y: target Series or None).
    """
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    # Drop identifier column if present
    if settings.ID_COLUMN in df.columns:
        df = df.drop(columns=[settings.ID_COLUMN])

    # Extract target if present
    y: Optional[pd.Series] = None
    if settings.TARGET_COLUMN in df.columns:
        y = (df[settings.TARGET_COLUMN].astype(str).str.strip().str.lower() == "yes").astype(int)
        df = df.drop(columns=[settings.TARGET_COLUMN])

    # Convert numeric-looking columns that may contain blank strings
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df, y


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Dynamically categorize features into numerical and categorical lists.

    Args:
        X: Cleaned feature DataFrame.

    Returns:
        Tuple of (numerical_features, categorical_features).
    """
    numerical_features: List[str] = []
    categorical_features: List[str] = []

    for col in X.columns:
        # SeniorCitizen is binary integer (0/1), treat as categorical/binary
        if col == "SeniorCitizen":
            categorical_features.append(col)
        elif pd.api.types.is_numeric_dtype(X[col]):
            numerical_features.append(col)
        else:
            categorical_features.append(col)

    return numerical_features, categorical_features


def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build a Scikit-Learn ColumnTransformer for leak-free feature preprocessing.

    - Numerical pipeline: SimpleImputer(strategy='median') + optional StandardScaler().
    - Categorical pipeline: SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore').

    Args:
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.
        scale_numeric: If True, applies StandardScaler to numerical columns.

    Returns:
        Configured ColumnTransformer instance.
    """
    # 1. Numerical Pipeline
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    # 2. Categorical Pipeline
    # handle_unknown='ignore' is critical for production: prevents crashes when new, unseen categories appear in inference
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 3. Assemble ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Retrieve output feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out is unavailable
        num_cols = preprocessor.transformers_[0][2]
        cat_encoder = preprocessor.transformers_[1][1].named_steps["onehot"]
        cat_cols = list(cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2]))
        return list(num_cols) + cat_cols


def prepare_and_split_data(
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    use_feature_engineering: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """Load raw dataset, clean features, and split into train/test sets using stratification.

    IMPORTANT: To prevent data leakage, the preprocessor is fitted strictly
    on X_train and only transforms X_test.

    Args:
        test_size: Proportion of data for test split (default from settings).
        random_state: Random seed for reproducibility (default from settings).
        use_feature_engineering: If True, applies CustomerFeatureEngineer domain features.

    Returns:
        Tuple of (X_train_df, X_test_df, y_train, y_test, fitted_preprocessor).
    """
    from src.feature_engineering import CustomerFeatureEngineer

    test_size = test_size if test_size is not None else settings.TEST_SIZE
    random_state = random_state if random_state is not None else settings.RANDOM_STATE

    # Step 1: Load and clean raw dataframe
    df_raw = load_raw_data()
    X_raw, y = clean_raw_dataframe(df_raw)

    if y is None:
        raise ValueError("Target column 'Churn' could not be found or extracted.")

    # Step 2: Stratified Train-Test Split (preserves 73.5% vs 26.5% class balance in both splits)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Step 3: Feature Engineering (if enabled)
    if use_feature_engineering:
        fe = CustomerFeatureEngineer()
        X_train_processed = fe.fit_transform(X_train_raw)
        X_test_processed = fe.transform(X_test_raw)
    else:
        X_train_processed = X_train_raw.copy()
        X_test_processed = X_test_raw.copy()

    # Step 4: Identify feature types from engineered training set
    num_cols, cat_cols = identify_feature_types(X_train_processed)

    # Step 5: Build ColumnTransformer
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=True)

    # Step 6: Fit preprocessor strictly on training data
    preprocessor.fit(X_train_processed)

    # Step 7: Transform train and test features
    feature_names = get_feature_names(preprocessor)
    X_train_arr = preprocessor.transform(X_train_processed)
    X_test_arr = preprocessor.transform(X_test_processed)

    X_train_df = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train_processed.index)
    X_test_df = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test_processed.index)

    return X_train_df, X_test_df, y_train, y_test, preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, output_path: Optional[Path] = None) -> Path:
    """Serialize the fitted preprocessor pipeline to disk via Joblib."""
    out_path = output_path or (settings.MODELS_DIR / "preprocessor.joblib")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, out_path)
    return out_path


def load_saved_preprocessor(model_path: Optional[Path] = None) -> ColumnTransformer:
    """Load a serialized preprocessor pipeline from disk."""
    path = model_path or (settings.MODELS_DIR / "preprocessor.joblib")
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {path}")
    return joblib.load(path)


def main() -> None:
    """Execution script for Phase 3 data preprocessing."""
    print("\n" + "=" * 65)
    print("      PHASE 3: DATA PREPROCESSING & LEAKAGE PREVENTION")
    print("=" * 65)

    X_train, X_test, y_train, y_test, preprocessor = prepare_and_split_data()

    print(f"[INFO] Train features shape: {X_train.shape} (Rows: {X_train.shape[0]:,}, Cols: {X_train.shape[1]})")
    print(f"[INFO] Test features shape : {X_test.shape} (Rows: {X_test.shape[0]:,}, Cols: {X_test.shape[1]})")
    print(f"[INFO] Train churn balance: {y_train.mean()*100:.2f}% Yes ({y_train.sum():,} / {len(y_train):,})")
    print(f"[INFO] Test churn balance : {y_test.mean()*100:.2f}% Yes ({y_test.sum():,} / {len(y_test):,})")

    # Verify no missing values exist in processed outputs
    train_nulls = X_train.isnull().sum().sum()
    test_nulls = X_test.isnull().sum().sum()
    print(f"[VERIFY] Remaining NaNs in Train Features: {train_nulls}")
    print(f"[VERIFY] Remaining NaNs in Test Features : {test_nulls}")

    # Serialize preprocessor pipeline artifact
    saved_path = save_preprocessor(preprocessor)
    print(f"[SUCCESS] Fitted preprocessor saved to: {saved_path}")

    # Save processed CSV for fast inspection & feature engineering
    processed_dir = settings.PROCESSED_DATA_PATH.parent
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_combined = X_train.copy()
    train_combined[settings.TARGET_COLUMN] = y_train
    train_combined.to_csv(settings.PROCESSED_DATA_PATH, index=False)
    print(f"[SUCCESS] Processed benchmark dataset written to: {settings.PROCESSED_DATA_PATH}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
