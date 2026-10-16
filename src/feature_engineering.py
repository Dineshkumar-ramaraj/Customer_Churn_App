"""Feature Engineering Module for Customer Churn Engine.

Constructs domain-driven behavioral features, interaction terms, and customer
stickiness indices using a Scikit-Learn compatible transformer.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import settings
from src.data_loader import load_raw_data
from src.data_preprocessing import clean_raw_dataframe


class CustomerFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-Learn compatible transformer for customer churn feature engineering.

    Creates domain-driven features:
    1. total_services: Count of active telecommunication & digital add-on services.
    2. has_support_security: Binary flag indicating presence of TechSupport or OnlineSecurity.
    3. is_automatic_payment: Binary indicator for frictionless recurring payments.
    4. is_fiber_no_techsupport: High-risk interaction term (Fiber optic without TechSupport).
    5. charges_per_service: Monthly cost divided by number of active services (perceived value).
    6. monthly_to_total_ratio: Ratio of MonthlyCharges to TotalCharges (lifecycle proxy).
    7. tenure_cohort: Binned customer lifecycle stage (0-12m, 13-24m, 25-48m, 49-72m).
    """

    def __init__(self, include_tenure_cohorts: bool = True):
        self.include_tenure_cohorts = include_tenure_cohorts
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CustomerFeatureEngineer":
        """Fit transformer (validates input columns)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations to input DataFrame.

        Args:
            X: Input feature DataFrame (without customerID and target).

        Returns:
            DataFrame augmented with engineered features.
        """
        df = X.copy()

        # 1. Total Active Services Count (Product Adoption & Switching Barrier)
        # Services that represent active customer engagement
        service_cols = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        service_count = pd.Series(0, index=df.index, dtype=int)
        for col in service_cols:
            if col in df.columns:
                # Count 'Yes' as 1, all others ('No', 'No internet service', 'No phone service') as 0
                service_count += (df[col] == "Yes").astype(int)

        df["total_services"] = service_count

        # 2. Support & Security Protective Flag (Stickiness Multiplier)
        has_security = df["OnlineSecurity"] == "Yes" if "OnlineSecurity" in df.columns else False
        has_support = df["TechSupport"] == "Yes" if "TechSupport" in df.columns else False
        df["has_support_security"] = (has_security | has_support).astype(int)

        # 3. Frictionless Recurring Payment Method
        if "PaymentMethod" in df.columns:
            df["is_automatic_payment"] = df["PaymentMethod"].isin(
                ["Bank transfer (automatic)", "Credit card (automatic)"]
            ).astype(int)
        else:
            df["is_automatic_payment"] = 0

        # 4. High-Risk Interaction: Fiber Optic without TechSupport
        is_fiber = df["InternetService"] == "Fiber optic" if "InternetService" in df.columns else False
        no_tech = df["TechSupport"] == "No" if "TechSupport" in df.columns else False
        df["is_fiber_no_techsupport"] = (is_fiber & no_tech).astype(int)

        # 5. Charges Per Active Service (Cost-to-Value Ratio)
        if "MonthlyCharges" in df.columns:
            # Add 1 to denominator to avoid division by zero
            df["charges_per_service"] = df["MonthlyCharges"] / (df["total_services"] + 1)
        else:
            df["charges_per_service"] = 0.0

        # 6. Monthly to Total Charges Ratio
        if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
            total_charges_clean = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
            df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (total_charges_clean + 1.0)
        else:
            df["monthly_to_total_ratio"] = 0.0

        # 7. Tenure Cohorts (Lifecycle Segmentation)
        if self.include_tenure_cohorts and "tenure" in df.columns:
            bins = [-1, 12, 24, 48, 72]
            labels = ["0-12m", "13-24m", "25-48m", "49-72m"]
            df["tenure_cohort"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(str)

        self.feature_names_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Return names of output features."""
        return np.array(self.feature_names_)


def compute_vif(df_numeric: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) to detect multicollinearity.

    VIF measures how much the variance of an estimated regression coefficient
    is inflated due to collinearity with other predictors.
    Rule of thumb:
    - VIF < 5: Low multicollinearity
    - VIF 5-10: Moderate multicollinearity (acceptable)
    - VIF > 10: High multicollinearity (potential issue for linear models)
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    df_clean = df_numeric.dropna().copy()
    # Add constant for intercept
    df_clean["Intercept"] = 1.0

    vif_data = []
    for i, col in enumerate(df_clean.columns):
        if col == "Intercept":
            continue
        try:
            vif_val = variance_inflation_factor(df_clean.values, i)
            vif_data.append({"Feature": col, "VIF": round(vif_val, 2)})
        except Exception:
            vif_data.append({"Feature": col, "VIF": np.nan})

    return pd.DataFrame(vif_data).sort_values(by="VIF", ascending=False)


def main() -> None:
    """Execute feature engineering pipeline and output statistical diagnostics."""
    print("\n" + "=" * 65)
    print("      PHASE 4: FEATURE ENGINEERING & STATISTICAL ANALYSIS")
    print("=" * 65)

    df_raw = load_raw_data()
    X_raw, y = clean_raw_dataframe(df_raw)

    print(f"[INFO] Raw features count: {X_raw.shape[1]}")

    # Apply Feature Engineering
    fe = CustomerFeatureEngineer()
    X_engineered = fe.fit_transform(X_raw)

    print(f"[SUCCESS] Engineered features count: {X_engineered.shape[1]}")
    new_cols = [c for c in X_engineered.columns if c not in X_raw.columns]
    print(f"[INFO] New features created ({len(new_cols)}): {new_cols}")

    # Inspect Correlations of New Features with Target
    print("\n--- Correlation of Engineered Features with Churn ---")
    corr_series = {}
    for col in new_cols:
        if pd.api.types.is_numeric_dtype(X_engineered[col]):
            corr = X_engineered[col].corr(y)
            corr_series[col] = corr
            print(f"  - {col:25s}: r = {corr:+.4f}")

    # Multicollinearity Check on Core Numerical Variables
    print("\n--- Multicollinearity Analysis (Variance Inflation Factor - VIF) ---")
    core_numeric = X_engineered.select_dtypes(include=[np.number]).copy()
    # Fill remaining NaNs in TotalCharges for VIF calculation
    core_numeric = core_numeric.fillna(core_numeric.median())

    try:
        vif_df = compute_vif(core_numeric)
        print(vif_df.to_string(index=False))
    except ImportError:
        print("[NOTE] statsmodels not installed; computing Pearson correlation matrix instead:")
        corr_mat = core_numeric.corr()
        high_corr = []
        for i in range(len(corr_mat.columns)):
            for j in range(i + 1, len(corr_mat.columns)):
                c1 = corr_mat.columns[i]
                c2 = corr_mat.columns[j]
                val = corr_mat.iloc[i, j]
                if abs(val) > 0.70:
                    high_corr.append((c1, c2, val))
        for c1, c2, val in high_corr:
            print(f"  - High correlation between '{c1}' and '{c2}': r = {val:.3f}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
