"""Unit tests for data validation module."""

import pytest
import pandas as pd
from src.data_validation import validate_raw_data


def test_empty_dataframe_raises_error():
    """Ensure an empty dataframe raises a ValueError."""
    df_empty = pd.DataFrame()
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_raw_data(df_empty)


def test_missing_required_column_raises_error():
    """Ensure a missing required column raises a ValueError."""
    df_sample = pd.DataFrame({
        "customerID": ["CUST-1"],
        "gender": ["Female"],
        "SeniorCitizen": [0],
        # Missing other required columns and Churn
    })
    with pytest.raises(ValueError, match="Schema validation failed"):
        validate_raw_data(df_sample)


def test_unexpected_target_values_raise_error():
    """Ensure invalid target values raise an error."""
    columns = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
    ]
    data = {col: ["valid"] for col in columns}
    data["Churn"] = ["InvalidTarget"]
    df_invalid_target = pd.DataFrame(data)

    with pytest.raises(ValueError, match="Unexpected target values"):
        validate_raw_data(df_invalid_target)
