"""Production Data Drift and Model Performance Monitoring Module.

Monitors population feature distribution shifts and target degradation using:
1. Population Stability Index (PSI).
2. Two-Sample Kolmogorov-Smirnov (KS) Test for continuous distributions.
3. Automated drift alert generation and audit report logging.
"""

from pathlib import Path
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import settings
from src.data_loader import load_raw_data
from src.data_preprocessing import clean_raw_dataframe


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between baseline and production distributions.

    Rule of Thumb:
    - PSI < 0.10: Stable (No shift).
    - 0.10 <= PSI < 0.25: Moderate shift (Warning: monitor and plan retraining).
    - PSI >= 0.25: Significant shift (Critical: trigger automated retraining).

    Args:
        expected: Reference/Baseline feature array (e.g., training set).
        actual: Current production feature array (e.g., recent inference batch).
        num_bins: Number of quantile buckets.
        epsilon: Smoothing constant to prevent division by zero or log(0).

    Returns:
        PSI float value.
    """
    # Remove NaNs
    exp_clean = expected[~np.isnan(expected)]
    act_clean = actual[~np.isnan(actual)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0

    # Determine quantile bins from reference distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(exp_clean, quantiles)
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0

    # Compute empirical bucket percentages
    exp_counts, _ = np.histogram(exp_clean, bins=bins)
    act_counts, _ = np.histogram(act_clean, bins=bins)

    exp_pct = exp_counts / len(exp_clean)
    act_pct = act_counts / len(act_clean)

    # Apply Laplace smoothing
    exp_pct = np.where(exp_pct == 0, epsilon, exp_pct)
    act_pct = np.where(act_pct == 0, epsilon, act_pct)

    # Compute PSI sum: sum((Actual - Expected) * ln(Actual / Expected))
    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_val)


def calculate_categorical_psi(
    expected_series: pd.Series,
    actual_series: pd.Series,
    epsilon: float = 1e-4,
) -> float:
    """Calculate PSI across discrete categories."""
    all_cats = list(set(expected_series.dropna().unique()).union(set(actual_series.dropna().unique())))
    if not all_cats:
        return 0.0

    exp_counts = expected_series.value_counts(normalize=True).to_dict()
    act_counts = actual_series.value_counts(normalize=True).to_dict()

    psi_total = 0.0
    for cat in all_cats:
        exp_p = exp_counts.get(cat, epsilon)
        act_p = act_counts.get(cat, epsilon)
        exp_p = max(exp_p, epsilon)
        act_p = max(act_p, epsilon)
        psi_total += (act_p - exp_p) * np.log(act_p / exp_p)

    return float(psi_total)


def audit_dataset_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Audit feature drift between baseline training data and a new batch.

    Args:
        reference_df: Baseline training feature DataFrame.
        current_df: Recent inference batch DataFrame.

    Returns:
        Structured drift report containing PSI, KS test statistics, and alert levels.
    """
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "reference_records": len(reference_df),
        "current_records": len(current_df),
        "features": {},
        "summary": {
            "total_features_audited": 0,
            "stable_features_count": 0,
            "warning_features_count": 0,
            "drifted_features_count": 0,
            "overall_status": "STABLE",
        },
    }

    common_cols = [c for c in reference_df.columns if c in current_df.columns and c not in ["customerID", "Churn"]]

    for col in common_cols:
        is_num = pd.api.types.is_numeric_dtype(reference_df[col])

        if is_num:
            exp_arr = pd.to_numeric(reference_df[col], errors="coerce").values
            act_arr = pd.to_numeric(current_df[col], errors="coerce").values

            psi = calculate_psi(exp_arr, act_arr)
            ks_stat, ks_pvalue = ks_2samp(exp_arr[~np.isnan(exp_arr)], act_arr[~np.isnan(act_arr)])

            # Alert level
            if psi < 0.10 and ks_pvalue > 0.05:
                status = "STABLE"
            elif psi < 0.25:
                status = "WARNING"
            else:
                status = "DRIFT_DETECTED"

            report["features"][col] = {
                "type": "numerical",
                "psi": round(psi, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pvalue), 4),
                "status": status,
            }
        else:
            psi = calculate_categorical_psi(reference_df[col], current_df[col])
            status = "STABLE" if psi < 0.10 else "WARNING" if psi < 0.25 else "DRIFT_DETECTED"

            report["features"][col] = {
                "type": "categorical",
                "psi": round(psi, 4),
                "status": status,
            }

        # Update summary counts
        report["summary"]["total_features_audited"] += 1
        if status == "STABLE":
            report["summary"]["stable_features_count"] += 1
        elif status == "WARNING":
            report["summary"]["warning_features_count"] += 1
        else:
            report["summary"]["drifted_features_count"] += 1

    # Overall system status
    if report["summary"]["drifted_features_count"] > 0:
        report["summary"]["overall_status"] = "CRITICAL_DRIFT"
    elif report["summary"]["warning_features_count"] > 0:
        report["summary"]["overall_status"] = "WARNING_MODERATE_SHIFT"
    else:
        report["summary"]["overall_status"] = "HEALTHY_STABLE"

    return report


def main() -> None:
    """Execute sample drift audit between train and test distributions."""
    print("\n" + "=" * 70)
    print("      PHASE 12: DATA DRIFT MONITORING AUDIT")
    print("=" * 70)

    from sklearn.model_selection import train_test_split
    df_raw = load_raw_data()
    X_raw, y = clean_raw_dataframe(df_raw)

    X_train, X_test, _, _ = train_test_split(
        X_raw,
        y,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=y,
    )

    print(f"[INFO] Auditing drift: Baseline Train ({len(X_train)}) vs Production Test ({len(X_test)})...")
    drift_report = audit_dataset_drift(X_train, X_test)

    summary = drift_report["summary"]
    print(f"\n[DRIFT AUDIT OUTCOME]: {summary['overall_status']}")
    print(f"  - Total Features Audited : {summary['total_features_audited']}")
    print(f"  - Stable Features        : {summary['stable_features_count']}")
    print(f"  - Warning Features       : {summary['warning_features_count']}")
    print(f"  - Drifted Features       : {summary['drifted_features_count']}")

    print("\n--- Feature PSI & Shift Metrics ---")
    for feat, data in list(drift_report["features"].items())[:8]:
        print(f"  - {feat:22s} | PSI: {data['psi']:.4f} | Status: {data['status']}")

    out_file = settings.EVALUATION_DIR / "drift_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(drift_report, f, indent=2)
    print(f"\n[SUCCESS] Drift report persisted to: {out_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
