"""Robustness, Edge-Case, and Reliability Test Suite for Customer Churn Engine.

Verifies system behavior under extreme conditions:
- Missing model artifacts.
- Out-of-distribution and unseen categorical values.
- Boundary numerical values (tenure=0, charges=0, tenure=120).
- Malformed strings in numeric columns ("N/A", "???", whitespace).
- Monotonicity and probability invariant checks.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from services.prediction_service import PredictionService
from app.api import app


@pytest.fixture
def service() -> PredictionService:
    """Fixture providing initialized PredictionService."""
    return PredictionService()


@pytest.fixture
def client() -> TestClient:
    """Fixture providing FastAPI TestClient."""
    return TestClient(app)


def test_missing_model_file_raises_filenotfound():
    """Verify that pointing PredictionService to a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        PredictionService(model_path=Path("models/non_existent_model_12345.joblib"))


def test_unseen_and_wildcard_categorical_values(service):
    """Verify pipeline gracefully handles unseen categories at inference without error."""
    unseen_customer = {
        "customerID": "WILD-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "Satellite Phone Line",  # Unseen category
        "InternetService": "Starlink Satellite",  # Unseen category
        "OnlineSecurity": "Quantum Shield",       # Unseen category
        "OnlineBackup": "Decentralized IPFS",     # Unseen category
        "DeviceProtection": "No",
        "TechSupport": "AI Bot Only",             # Unseen category
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Lifetime Subscription",      # Unseen category
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Cryptocurrency Bitcoin", # Unseen category
        "MonthlyCharges": 75.0,
        "TotalCharges": 900.0,
    }

    result = service.predict_single(unseen_customer)

    assert result["customer_id"] == "WILD-001"
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["prediction"] in ["Yes", "No"]
    assert result["risk_level"] in ["Low", "Medium", "High"]


def test_extreme_numerical_boundary_values(service):
    """Verify pipeline handles extreme valid boundaries (tenure=0, charges=0, tenure=120)."""
    boundary_cases = [
        # Zero tenure, zero charges (brand new account)
        {"tenure": 0, "MonthlyCharges": 0.0, "TotalCharges": 0.0},
        # High tenure, high charges (10-year enterprise user)
        {"tenure": 120, "MonthlyCharges": 500.0, "TotalCharges": 60000.0},
        # Zero total charges with positive monthly charge
        {"tenure": 1, "MonthlyCharges": 100.0, "TotalCharges": " "},
    ]

    base_customer = {
        "customerID": "BOUND-CUST",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
    }

    for b_case in boundary_cases:
        cust = {**base_customer, **b_case}
        res = service.predict_single(cust)
        assert 0.0 <= res["churn_probability"] <= 1.0
        assert not np.isnan(res["churn_probability"])


def test_corrupt_total_charges_strings(service):
    """Verify non-numeric corrupt strings in TotalCharges coerce gracefully to NaN without error."""
    corrupt_values = ["", " ", "N/A", "null", "$$$$", "unknown", "1,234.50"]
    base_customer = {
        "customerID": "CORRUPT-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
    }

    for val in corrupt_values:
        cust = {**base_customer, "TotalCharges": val}
        res = service.predict_single(cust)
        assert 0.0 <= res["churn_probability"] <= 1.0


def test_monotonicity_sanity_check(service):
    """Monotonicity check: High-risk profile MUST have higher probability than low-risk profile."""
    high_risk_profile = {
        "customerID": "HIGH-MOTO",
        "gender": "Male",
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
        "MonthlyCharges": 105.0,
        "TotalCharges": 105.0,
    }

    # Identical customer, but with 2-year contract, low monthly bill, tech support, and auto payment
    low_risk_profile = {
        **high_risk_profile,
        "customerID": "LOW-MOTO",
        "tenure": 60,
        "Contract": "Two year",
        "TechSupport": "Yes",
        "OnlineSecurity": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 35.0,
        "TotalCharges": 2100.0,
    }

    res_high = service.predict_single(high_risk_profile)
    res_low = service.predict_single(low_risk_profile)

    assert res_high["churn_probability"] > res_low["churn_probability"]
    # The probability gap should be substantial (> 0.40)
    assert (res_high["churn_probability"] - res_low["churn_probability"]) > 0.40


def test_api_empty_batch_rejected(client):
    """Verify that POST /predict/batch with empty customer list returns HTTP 422."""
    empty_payload = {"customers": []}
    response = client.post("/predict/batch", json=empty_payload)
    assert response.status_code == 422


def test_api_out_of_range_numerical_rejected(client):
    """Verify that passing out-of-range numerical values to API triggers HTTP 422."""
    # Negative tenure
    bad_payload_1 = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": -1,  # Invalid: ge=0
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 50.0,
    }
    resp1 = client.post("/predict", json=bad_payload_1)
    assert resp1.status_code == 422

    # Excessive MonthlyCharges (> 500)
    bad_payload_2 = {**bad_payload_1, "tenure": 10, "MonthlyCharges": 1200.0}
    resp2 = client.post("/predict", json=bad_payload_2)
    assert resp2.status_code == 422
