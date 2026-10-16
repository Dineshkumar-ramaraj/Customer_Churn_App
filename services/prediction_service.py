"""Prediction Service Module for Customer Churn Engine.

Decoupled inference engine that accepts individual or batch customer records,
applies the serialized end-to-end pipeline, assigns calibrated risk tiers,
identifies primary risk drivers, and formats production-ready responses.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import joblib

from src.config import settings
from src.data_preprocessing import clean_raw_dataframe


class PredictionService:
    """Thread-safe prediction service for real-time and batch customer churn scoring."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        default_threshold: Optional[float] = None,
    ):
        self.model_path = model_path or (settings.MODELS_DIR / "best_churn_model.joblib")
        self.metadata_path = metadata_path or (settings.MODELS_DIR / "model_metadata.json")
        self.pipeline = None
        self.metadata = {}
        self.decision_threshold = default_threshold

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load serialized pipeline and metadata from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{self.model_path}'. "
                "Ensure Phase 5 model training has been executed."
            )
        self.pipeline = joblib.load(self.model_path)

        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}

        # Set decision threshold from metadata if not explicitly provided
        if self.decision_threshold is None:
            self.decision_threshold = self.metadata.get(
                "optimal_decision_threshold",
                settings.DEFAULT_DECISION_THRESHOLD,
            )

    @staticmethod
    def assign_risk_level(
        probability: float,
        low_cutoff: float = 0.30,
        high_cutoff: float = 0.60,
    ) -> str:
        """Assign configurable business risk tier based on predicted probability.

        Args:
            probability: Churn probability score between 0.0 and 1.0.
            low_cutoff: Upper bound for Low Risk tier.
            high_cutoff: Lower bound for High Risk tier.

        Returns:
            Risk tier string ('Low', 'Medium', or 'High').
        """
        if probability < low_cutoff:
            return "Low"
        elif probability < high_cutoff:
            return "Medium"
        else:
            return "High"

    @staticmethod
    def identify_customer_risk_factors(customer_dict: Dict[str, Any]) -> List[str]:
        """Explainable risk drivers: highlight high-risk attributes for a given customer."""
        risk_factors = []

        # Contract commitment
        contract = customer_dict.get("Contract", "")
        if contract == "Month-to-month":
            risk_factors.append("Month-to-month contract (High churn susceptibility)")

        # Early tenure
        try:
            tenure = float(customer_dict.get("tenure", 0))
            if tenure <= 12:
                risk_factors.append(f"Early tenure stage ({int(tenure)} months) - within first-year churn window")
        except (ValueError, TypeError):
            pass

        # Internet service & TechSupport combination
        internet = customer_dict.get("InternetService", "")
        tech_support = customer_dict.get("TechSupport", "")
        if internet == "Fiber optic" and tech_support == "No":
            risk_factors.append("Fiber optic subscriber without active Technical Support")

        # Payment method
        payment = customer_dict.get("PaymentMethod", "")
        if payment == "Electronic check":
            risk_factors.append("Manual Electronic check payment (Lacks auto-renew friction barrier)")

        # Online security
        online_sec = customer_dict.get("OnlineSecurity", "")
        if online_sec == "No" and internet != "No":
            risk_factors.append("No Online Security add-on")

        # High charges per service
        try:
            monthly = float(customer_dict.get("MonthlyCharges", 0.0))
            if monthly > 85.0:
                risk_factors.append(f"Elevated monthly charge (${monthly:.2f}/mo)")
        except (ValueError, TypeError):
            pass

        return risk_factors if risk_factors else ["No major high-risk indicators detected"]

    def predict_single(
        self,
        customer_data: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Score a single customer record and return structured prediction payload.

        Args:
            customer_data: Dictionary containing customer features.
            threshold: Optional custom decision threshold overriding default.

        Returns:
            Dictionary matching the target production response schema.
        """
        thresh = threshold if threshold is not None else self.decision_threshold

        # Extract customer identifier if present
        customer_id = customer_data.get("customerID") or customer_data.get("customer_id") or "UNKNOWN_CUST"

        # Create single-row DataFrame
        df_row = pd.DataFrame([customer_data])

        # Isolate identifier
        if "customerID" in df_row.columns:
            df_row = df_row.drop(columns=["customerID"])
        if "customer_id" in df_row.columns:
            df_row = df_row.drop(columns=["customer_id"])

        # Coerce TotalCharges
        if "TotalCharges" in df_row.columns:
            df_row["TotalCharges"] = pd.to_numeric(df_row["TotalCharges"], errors="coerce")

        # Run pipeline inference
        prob_churn = float(self.pipeline.predict_proba(df_row)[0, 1])
        prediction_label = "Yes" if prob_churn >= thresh else "No"
        risk_level = self.assign_risk_level(prob_churn)
        risk_factors = self.identify_customer_risk_factors(customer_data)

        return {
            "customer_id": str(customer_id),
            "churn_probability": round(prob_churn, 4),
            "prediction": prediction_label,
            "risk_level": risk_level,
            "decision_threshold_applied": round(thresh, 4),
            "model_version": self.metadata.get("model_name", "Tuned XGBoost 1.0.0"),
            "key_risk_factors": risk_factors,
        }

    def predict_batch(
        self,
        df_batch: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Score a batch DataFrame of customer records.

        Args:
            df_batch: DataFrame containing batch customer features.
            threshold: Optional custom decision threshold.

        Returns:
            DataFrame with original columns plus prediction outputs.
        """
        thresh = threshold if threshold is not None else self.decision_threshold
        df = df_batch.copy()

        # Isolate identifier column
        id_series = None
        if "customerID" in df.columns:
            id_series = df["customerID"]
            df_features = df.drop(columns=["customerID"])
        elif "customer_id" in df.columns:
            id_series = df["customer_id"]
            df_features = df.drop(columns=["customer_id"])
        else:
            id_series = pd.Series([f"ROW-{i:04d}" for i in range(len(df))], index=df.index)
            df_features = df.copy()

        # Drop ground-truth target if present in historical evaluation uploads
        if "Churn" in df_features.columns:
            df_features = df_features.drop(columns=["Churn"])
        if "churn" in df_features.columns:
            df_features = df_features.drop(columns=["churn"])

        # Coerce TotalCharges to numeric
        if "TotalCharges" in df_features.columns:
            df_features["TotalCharges"] = pd.to_numeric(df_features["TotalCharges"], errors="coerce")

        # Vectorized batch prediction
        probabilities = self.pipeline.predict_proba(df_features)[:, 1]
        predictions = ["Yes" if p >= thresh else "No" for p in probabilities]
        risk_levels = [self.assign_risk_level(p) for p in probabilities]

        output_df = df.copy()
        output_df["customerID"] = id_series
        output_df["churn_probability"] = np.round(probabilities, 4)
        output_df["churn_prediction"] = predictions
        output_df["risk_level"] = risk_levels

        return output_df


# Global singleton instance
prediction_service = PredictionService()
