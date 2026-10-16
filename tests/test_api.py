"""Automated integration tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.api import app


@pytest.fixture
def client() -> TestClient:
    """Fixture providing TestClient instance."""
    return TestClient(app)


@pytest.fixture
def valid_customer_payload() -> dict:
    """Valid customer input dictionary conforming to CustomerInputSchema."""
    return {
        "customerID": "API-TEST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
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
        "MonthlyCharges": 92.50,
        "TotalCharges": 277.50,
    }


def test_health_endpoint(client):
    """Test GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data


def test_model_info_endpoint(client):
    """Test GET /model-info returns valid metadata and threshold."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "optimal_decision_threshold" in data
    assert data["optimal_decision_threshold"] == 0.54
    assert "test_metrics" in data


def test_predict_single_valid(client, valid_customer_payload):
    """Test POST /predict with a valid payload returns 200 and structured prediction."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "API-TEST-001"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["prediction"] in ["Yes", "No"]
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert isinstance(data["key_risk_factors"], list)
    assert len(data["key_risk_factors"]) > 0


def test_predict_single_validation_error(client, valid_customer_payload):
    """Test POST /predict with invalid field values triggers 422 Unprocessable Entity."""
    invalid_payload = valid_customer_payload.copy()
    invalid_payload["tenure"] = -5  # Negative tenure violates ge=0 constraint
    invalid_payload["MonthlyCharges"] = -10.0  # Violates ge=0 constraint

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_batch_endpoint(client, valid_customer_payload):
    """Test POST /predict/batch scores multiple customer records simultaneously."""
    cust2 = valid_customer_payload.copy()
    cust2["customerID"] = "API-TEST-002"
    cust2["Contract"] = "Two year"
    cust2["tenure"] = 60

    batch_payload = {
        "customers": [valid_customer_payload, cust2],
        "custom_threshold": 0.54,
    }

    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["customer_id"] == "API-TEST-001"
    assert data["predictions"][1]["customer_id"] == "API-TEST-002"


def test_metrics_endpoint(client, valid_customer_payload):
    """Test GET /metrics returns accurate operational metrics."""
    # Score a single prediction to increment tracker
    client.post("/predict", json=valid_customer_payload)

    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_single_predictions_served"] >= 1
    assert data["total_records_scored"] >= 1
    assert data["service_uptime_seconds"] >= 0.0
