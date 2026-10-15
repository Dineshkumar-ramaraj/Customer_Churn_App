"""Data validation module for raw customer churn dataset.

Validates schema integrity, data types, missing values, duplicates,
and target variable distribution before downstream processing.
"""

from typing import Dict, Any, List
import pandas as pd
from src.config import settings


def validate_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate raw customer churn dataset structure and schema.

    Args:
        df: Pandas DataFrame containing the loaded raw dataset.

    Returns:
        Dict containing validation status and diagnostic details.

    Raises:
        ValueError: If essential validation checks fail (e.g. empty dataset, missing target).
    """
    if df.empty:
        raise ValueError("Data validation failed: Dataset is empty.")

    validation_results: Dict[str, Any] = {
        "is_valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "missing_expected_columns": [],
        "extra_columns": [],
        "null_counts": {},
        "blank_string_counts": {},
        "target_distribution": {},
        "duplicate_rows": int(df.duplicated().sum()),
        "id_unique": True,
    }

    # 1. Schema check
    expected_set = set(settings.EXPECTED_COLUMNS)
    actual_set = set(df.columns)

    missing_cols = list(expected_set - actual_set)
    extra_cols = list(actual_set - expected_set)

    validation_results["missing_expected_columns"] = missing_cols
    validation_results["extra_columns"] = extra_cols

    if missing_cols:
        validation_results["is_valid"] = False
        raise ValueError(f"Schema validation failed. Missing expected columns: {missing_cols}")

    # 2. Target column check
    if settings.TARGET_COLUMN not in df.columns:
        validation_results["is_valid"] = False
        raise ValueError(f"Target column '{settings.TARGET_COLUMN}' not found in dataset.")

    target_counts = df[settings.TARGET_COLUMN].value_counts(dropna=False).to_dict()
    validation_results["target_distribution"] = target_counts

    # Validate target values are strictly Yes/No or binary
    valid_targets = {"Yes", "No", 1, 0, "1", "0"}
    actual_targets = set(df[settings.TARGET_COLUMN].dropna().unique())
    if not actual_targets.issubset(valid_targets):
        validation_results["is_valid"] = False
        raise ValueError(f"Unexpected target values found in '{settings.TARGET_COLUMN}': {actual_targets}")

    # 3. Customer ID Uniqueness check
    if settings.ID_COLUMN in df.columns:
        id_unique = df[settings.ID_COLUMN].is_unique
        validation_results["id_unique"] = id_unique
        if not id_unique:
            dup_count = len(df) - df[settings.ID_COLUMN].nunique()
            print(f"[WARNING] customerID contains {dup_count} non-unique values.")

    # 4. Null & Blank string checks
    # In Telco Churn, TotalCharges notoriously contains blank spaces ' ' for tenure=0 customers
    nulls = df.isnull().sum()
    validation_results["null_counts"] = {col: int(cnt) for col, cnt in nulls.items() if cnt > 0}

    blank_counts = {}
    str_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    for col in str_cols:
        blanks = (df[col].astype(str).str.strip() == "").sum()
        if blanks > 0:
            blank_counts[col] = int(blanks)
    validation_results["blank_string_counts"] = blank_counts

    return validation_results


def print_validation_report(val_info: Dict[str, Any]) -> None:
    """Print a clean, human-readable summary of the validation report."""
    print("=" * 60)
    print("        RAW DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Validation Status    : {'PASSED' if val_info['is_valid'] else 'FAILED'}")
    print(f"Total Rows           : {val_info['row_count']:,}")
    print(f"Total Columns        : {val_info['column_count']}")
    print(f"Duplicate Rows       : {val_info['duplicate_rows']}")
    print(f"Customer IDs Unique  : {val_info['id_unique']}")
    print("-" * 60)
    print("Target Variable Distribution:")
    for label, count in val_info["target_distribution"].items():
        pct = (count / val_info["row_count"]) * 100
        print(f"  - '{label}': {count:,} ({pct:.2f}%)")
    print("-" * 60)

    if val_info["null_counts"]:
        print("Explicit Null Values Detected:")
        for col, count in val_info["null_counts"].items():
            print(f"  - {col}: {count}")
    else:
        print("Explicit Null Values: None (0)")

    if val_info["blank_string_counts"]:
        print("Whitespace/Blank Strings Detected (Hidden Missing Values):")
        for col, count in val_info["blank_string_counts"].items():
            print(f"  - {col}: {count} blank entries")
    else:
        print("Blank String Entries: None (0)")
    print("=" * 60)
