"""Unit tests for PredictionService inference module."""

import pytest
import pandas as pd
from services.prediction_service import PredictionService


@pytest.fixture
def service() -> PredictionService:
    """Fixture providing initialized PredictionService instance."""
    return PredictionService()


@pytest.fixture
def high_risk_customer() -> dict:
    """Attributes characteristic of a high-risk churn customer."""
    return {
        "customerID": "TEST-HIGH-01",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": "95.50",
    }


@pytest.fixture
def low_risk_customer() -> dict:
    """Attributes characteristic of a loyal, low-risk customer."""
    return {
        "customerID": "TEST-LOW-01",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 60.20,
        "TotalCharges": "3913.00",
    }


def test_predict_single_high_risk(service, high_risk_customer):
    """Ensure high-risk customer receives elevated probability and High risk tier."""
    res = service.predict_single(high_risk_customer)

    assert res["customer_id"] == "TEST-HIGH-01"
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["churn_probability"] > 0.50
    assert res["prediction"] == "Yes"
    assert res["risk_level"] in ["Medium", "High"]
    assert len(res["key_risk_factors"]) > 0


def test_predict_single_low_risk(service, low_risk_customer):
    """Ensure low-risk customer receives low probability and No churn prediction."""
    res = service.predict_single(low_risk_customer)

    assert res["customer_id"] == "TEST-LOW-01"
    assert res["churn_probability"] < 0.35
    assert res["prediction"] == "No"
    assert res["risk_level"] == "Low"


def test_predict_batch_shape_and_columns(service, high_risk_customer, low_risk_customer):
    """Verify batch scoring appends probability and prediction columns correctly."""
    df_batch = pd.DataFrame([high_risk_customer, low_risk_customer])
    df_scored = service.predict_batch(df_batch)

    assert len(df_scored) == 2
    assert "churn_probability" in df_scored.columns
    assert "churn_prediction" in df_scored.columns
    assert "risk_level" in df_scored.columns
    assert df_scored.loc[0, "churn_prediction"] == "Yes"
    assert df_scored.loc[1, "churn_prediction"] == "No"
