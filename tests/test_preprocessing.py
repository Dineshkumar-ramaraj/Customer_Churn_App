"""Unit tests for data preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import (
    clean_raw_dataframe,
    identify_feature_types,
    build_preprocessor,
    prepare_and_split_data,
    save_preprocessor,
    load_saved_preprocessor,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Provide a minimal test DataFrame mimicking the raw Telco dataset."""
    return pd.DataFrame({
        "customerID": ["CUST-001", "CUST-002", "CUST-003", "CUST-004"],
        "gender": ["Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0],
        "Partner": ["Yes", "No", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No"],
        "tenure": [1, 34, 2, 45],
        "PhoneService": ["No", "Yes", "Yes", "No"],
        "MultipleLines": ["No phone service", "No", "No", "No phone service"],
        "InternetService": ["DSL", "DSL", "DSL", "DSL"],
        "OnlineSecurity": ["No", "Yes", "Yes", "Yes"],
        "OnlineBackup": ["Yes", "No", "Yes", "No"],
        "DeviceProtection": ["No", "Yes", "No", "Yes"],
        "TechSupport": ["No", "No", "No", "Yes"],
        "StreamingTV": ["No", "No", "No", "No"],
        "StreamingMovies": ["No", "No", "No", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Mailed check", "Bank transfer"],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
        "TotalCharges": ["29.85", "1889.5", " ", "1840.75"],  # Notice the whitespace string
        "Churn": ["No", "No", "Yes", "No"],
    })


def test_clean_raw_dataframe_drops_id_and_coerces_blanks(sample_raw_df):
    """Test that customerID is removed and TotalCharges blank is converted to NaN."""
    X, y = clean_raw_dataframe(sample_raw_df)

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert y is not None
    assert len(y) == 4
    assert list(y) == [0, 0, 1, 0]

    # Verify whitespace converted to NaN
    assert np.isnan(X.loc[2, "TotalCharges"])
    assert X["TotalCharges"].dtype == np.float64


def test_preprocessor_handles_unseen_categories(sample_raw_df):
    """Test that preprocessor handles unknown categorical levels at inference time without error."""
    X, _ = clean_raw_dataframe(sample_raw_df)
    num_cols, cat_cols = identify_feature_types(X)

    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor.fit(X)

    # Create unseen inference record
    inference_df = X.iloc[[0]].copy()
    inference_df.loc[0, "PaymentMethod"] = "Crypto Payment"  # Unseen category
    inference_df.loc[0, "Contract"] = "Decade Contract"     # Unseen category

    # Transform should not raise ValueError thanks to handle_unknown='ignore'
    transformed = preprocessor.transform(inference_df)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_pipeline_zero_nulls_output():
    """Verify that end-to-end prepare_and_split_data produces clean matrices with no NaNs."""
    X_train, X_test, y_train, y_test, preprocessor = prepare_and_split_data()

    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
    assert len(X_train) + len(X_test) == 7043
    assert len(y_train) + len(y_test) == 7043
