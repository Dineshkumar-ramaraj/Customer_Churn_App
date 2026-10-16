"""Unit tests for data drift monitoring module."""

import pytest
import numpy as np
import pandas as pd
from src.monitoring import calculate_psi, calculate_categorical_psi, audit_dataset_drift


def test_psi_identical_distributions_near_zero():
    """Verify PSI of identical distributions evaluates to near zero."""
    np.random.seed(42)
    dist = np.random.normal(loc=50, scale=15, size=2000)
    psi = calculate_psi(dist, dist)
    assert psi < 0.02


def test_psi_shifted_distributions_high():
    """Verify PSI flags substantial distribution shifts (> 0.25)."""
    np.random.seed(42)
    dist_base = np.random.normal(loc=50, scale=10, size=2000)
    # Significant shift in mean and variance
    dist_shifted = np.random.normal(loc=75, scale=20, size=2000)

    psi = calculate_psi(dist_base, dist_shifted)
    assert psi > 0.25


def test_audit_dataset_drift_structure():
    """Verify drift audit dictionary schema."""
    df_base = pd.DataFrame({
        "tenure": np.random.randint(1, 72, 100),
        "MonthlyCharges": np.random.uniform(20, 100, 100),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], 100),
    })
    df_curr = df_base.copy()

    report = audit_dataset_drift(df_base, df_curr)
    assert "summary" in report
    assert report["summary"]["overall_status"] == "HEALTHY_STABLE"
    assert "tenure" in report["features"]
    assert "Contract" in report["features"]
