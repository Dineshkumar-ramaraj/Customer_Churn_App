"""Unit tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import CustomerFeatureEngineer


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Sample DataFrame with standard Telco customer attributes."""
    return pd.DataFrame({
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [2, 48],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["Yes", "No"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Credit card (automatic)"],
        "MonthlyCharges": [95.0, 65.0],
        "TotalCharges": [190.0, 3120.0],
    })


def test_customer_feature_engineer_output_columns(sample_feature_df):
    """Test that all expected engineered features are properly generated."""
    fe = CustomerFeatureEngineer()
    transformed = fe.fit_transform(sample_feature_df)

    expected_new_cols = [
        "total_services",
        "has_support_security",
        "is_automatic_payment",
        "is_fiber_no_techsupport",
        "charges_per_service",
        "monthly_to_total_ratio",
        "tenure_cohort",
    ]

    for col in expected_new_cols:
        assert col in transformed.columns, f"Expected engineered feature '{col}' missing."


def test_feature_engineering_logic_values(sample_feature_df):
    """Verify precision of calculations for engineered metrics."""
    fe = CustomerFeatureEngineer()
    df_out = fe.fit_transform(sample_feature_df)

    # Row 0: PhoneService=Yes(1), StreamingTV=Yes(1), StreamingMovies=Yes(1), rest No -> total_services = 3
    assert df_out.loc[0, "total_services"] == 3
    # Row 0: Fiber optic + No TechSupport -> is_fiber_no_techsupport = 1
    assert df_out.loc[0, "is_fiber_no_techsupport"] == 1
    # Row 0: Electronic check -> is_automatic_payment = 0
    assert df_out.loc[0, "is_automatic_payment"] == 0
    # Row 0: TechSupport=No & OnlineSecurity=No -> has_support_security = 0
    assert df_out.loc[0, "has_support_security"] == 0
    # Row 0: charges_per_service = 95.0 / (3 + 1) = 23.75
    assert df_out.loc[0, "charges_per_service"] == 23.75

    # Row 1: Credit card (automatic) -> is_automatic_payment = 1
    assert df_out.loc[1, "is_automatic_payment"] == 1
    # Row 1: DSL -> is_fiber_no_techsupport = 0
    assert df_out.loc[1, "is_fiber_no_techsupport"] == 0
    # Row 1: TechSupport=Yes -> has_support_security = 1
    assert df_out.loc[1, "has_support_security"] == 1


def test_zero_division_resilience():
    """Ensure pipeline handles extreme edge cases (0 charges, 0 services) without errors."""
    df_edge = pd.DataFrame({
        "tenure": [0],
        "MonthlyCharges": [0.0],
        "TotalCharges": [0.0],
        "InternetService": ["No"],
        "TechSupport": ["No internet service"],
        "OnlineSecurity": ["No internet service"],
        "PaymentMethod": ["Mailed check"],
    })

    fe = CustomerFeatureEngineer()
    df_out = fe.fit_transform(df_edge)

    assert not np.isinf(df_out["charges_per_service"]).any()
    assert not np.isinf(df_out["monthly_to_total_ratio"]).any()
    assert df_out.loc[0, "charges_per_service"] == 0.0
