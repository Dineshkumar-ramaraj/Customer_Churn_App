"""Tests for Business Value, ROI Modeling, and Retention Strategy."""

import numpy as np
import pytest

from src.business_value import (
    BusinessParameters,
    calculate_cost_benefit,
    compute_roi_curve,
    generate_ab_test_specification,
    segment_customer_retention_tiers,
)


def test_business_parameters_ltv_calculation():
    """Verify Customer Lifetime Value formula."""
    params = BusinessParameters(
        avg_monthly_charges=100.0,
        customer_lifetime_months=20,
        gross_margin_rate=0.80,
        intervention_cost=50.0,
        intervention_success_rate=0.25,
    )
    # LTV = 100 * 20 * 0.80 = 1600.0
    assert params.customer_ltv == 1600.0


def test_calculate_cost_benefit():
    """Verify cost benefit matrix and financial returns."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 1])
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.85])
    threshold = 0.50

    params = BusinessParameters(
        avg_monthly_charges=50.0,
        customer_lifetime_months=10,
        gross_margin_rate=1.0,  # LTV = 500
        intervention_cost=20.0,
        intervention_success_rate=0.50,
    )

    cb = calculate_cost_benefit(y_true, y_prob, threshold, params)

    # Threshold 0.50 flags: indices 0 (1), 1 (1), 2 (1), 3 (0), 9 (1) -> 4 TP, 1 FP
    assert cb["confusion_matrix"]["tp"] == 4
    assert cb["confusion_matrix"]["fp"] == 1
    assert cb["total_interventions"] == 5

    # Cost: 5 * 20 = 100
    assert cb["total_intervention_cost"] == 100.0

    # Churners saved: 4 * 0.50 = 2.0
    assert cb["churners_saved"] == 2.0

    # Preserved value: 2.0 * 500 = 1000.0
    assert cb["gross_preserved_value"] == 1000.0

    # Net benefit: 1000 - 100 = 900.0
    assert cb["net_financial_benefit"] == 900.0

    # ROI %: (900 / 100) * 100 = 900%
    assert cb["roi_percentage"] == 900.0


def test_compute_roi_curve():
    """Verify ROI curve returns monotonic thresholds."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_prob = np.array([0.9, 0.8, 0.4, 0.2, 0.1])
    thresholds = np.array([0.2, 0.5, 0.8])

    df_roi = compute_roi_curve(y_true, y_prob, thresholds=thresholds)
    assert len(df_roi) == 3
    assert "net_financial_benefit" in df_roi.columns
    assert "roi_pct" in df_roi.columns


def test_segment_customer_retention_tiers():
    """Verify risk tier assignment logic."""
    import pandas as pd

    test_df = pd.DataFrame({
        "churn_probability": [0.85, 0.75, 0.60, 0.45, 0.15],
        "MonthlyCharges": [95.0, 35.0, 65.0, 50.0, 20.0],
    })

    segmented = segment_customer_retention_tiers(test_df)
    assert segmented.iloc[0]["retention_tier"] == "Tier 1: VIP High-Risk"
    assert segmented.iloc[1]["retention_tier"] == "Tier 2: Standard High-Risk"
    assert segmented.iloc[2]["retention_tier"] == "Tier 3: Moderate Risk"
    assert segmented.iloc[3]["retention_tier"] == "Tier 4: Early Watchlist"
    assert segmented.iloc[4]["retention_tier"] == "Tier 5: Loyal / Healthy"


def test_generate_ab_test_specification():
    """Verify A/B testing specification structure."""
    spec = generate_ab_test_specification()
    assert "experiment_name" in spec
    assert "statistical_parameters" in spec
    assert spec["statistical_parameters"]["statistical_power"] == 0.80
    assert spec["statistical_parameters"]["significance_level_alpha"] == 0.05
