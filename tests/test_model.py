"""Unit tests for model training and pipeline serialization."""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.train import get_model_candidates, build_full_pipeline
from src.data_preprocessing import clean_raw_dataframe


@pytest.fixture
def small_train_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provide a minimal synthetic dataset to verify training pipelines."""
    data = {
        "customerID": [f"CUST-{i}" for i in range(20)],
        "gender": ["Male", "Female"] * 10,
        "SeniorCitizen": [0, 1] * 10,
        "Partner": ["Yes", "No"] * 10,
        "Dependents": ["No", "Yes"] * 10,
        "tenure": [1, 24, 12, 60] * 5,
        "PhoneService": ["Yes", "No"] * 10,
        "MultipleLines": ["No", "Yes"] * 10,
        "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic"] * 5,
        "OnlineSecurity": ["No", "Yes"] * 10,
        "OnlineBackup": ["Yes", "No"] * 10,
        "DeviceProtection": ["No", "Yes"] * 10,
        "TechSupport": ["No", "Yes"] * 10,
        "StreamingTV": ["No", "Yes"] * 10,
        "StreamingMovies": ["No", "Yes"] * 10,
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"] * 5,
        "PaperlessBilling": ["Yes", "No"] * 10,
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] * 5,
        "MonthlyCharges": [70.5, 45.0, 20.0, 95.0] * 5,
        "TotalCharges": ["70.5", "1080.0", " ", "5700.0"] * 5,
        "Churn": ["Yes", "No", "No", "Yes"] * 5,
    }
    df = pd.DataFrame(data)
    X, y = clean_raw_dataframe(df)
    return X, y


def test_model_candidates_initialization():
    """Verify all 3 candidate models are configured properly."""
    models = get_model_candidates(scale_pos_weight=2.5)
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    assert "XGBoost" in models


def test_pipeline_fit_predict_probabilities(small_train_data):
    """Verify end-to-end pipeline outputs valid probability arrays between 0 and 1."""
    X, y = small_train_data
    models = get_model_candidates(scale_pos_weight=1.0)
    pipeline = build_full_pipeline(models["Logistic Regression"], X)

    # Fit pipeline
    pipeline.fit(X, y)

    # Predict probabilities on raw features
    probs = pipeline.predict_proba(X)
    assert probs.shape == (20, 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    # Probabilities must sum to 1.0
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(20), atol=1e-5)


def test_evaluate_single_model_and_threshold(small_train_data):
    """Verify evaluation calculations and threshold tuning logic."""
    from src.evaluate import evaluate_single_model, find_optimal_threshold

    X, y = small_train_data
    models = get_model_candidates(scale_pos_weight=1.0)
    pipeline = build_full_pipeline(models["Logistic Regression"], X)
    pipeline.fit(X, y)

    res = evaluate_single_model(pipeline, X, y, threshold=0.50)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["recall"] <= 1.0
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert res["confusion_matrix"] is not None

    best_thresh, df_thresh = find_optimal_threshold(y, res["y_prob"], metric="f1")
    assert 0.10 <= best_thresh <= 0.90
    assert len(df_thresh) == 81
