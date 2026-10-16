"""Command-Line Interface (CLI) for Customer Churn Inference.

Provides single-customer interactive prediction and batch CSV scoring
using the centralized PredictionService.
"""

import sys
import argparse
from pathlib import Path
import json
import pandas as pd

from src.config import settings
from services.prediction_service import prediction_service


def score_single_customer_sample() -> None:
    """Demonstrate single-customer real-time scoring with sample realistic attributes."""
    print("\n--- Scoring Sample Customer Record ---")
    sample_customer = {
        "customerID": "DEMO-7890",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
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
        "TotalCharges": "185.00",
    }

    result = prediction_service.predict_single(sample_customer)
    print(json.dumps(result, indent=2))


def score_batch_file(input_csv: Path, output_csv: Path) -> None:
    """Score an entire CSV file of customer records and export results."""
    print(f"\n[INFO] Loading batch dataset from: {input_csv}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found at: {input_csv}")

    df_in = pd.read_csv(input_csv)
    print(f"[INFO] Scored records count: {len(df_in):,}")

    df_scored = prediction_service.predict_batch(df_in)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Batch predictions exported to: {output_csv}")

    # Display summary
    churn_counts = df_scored["churn_prediction"].value_counts()
    risk_counts = df_scored["risk_level"].value_counts()
    print("\n--- Batch Scoring Summary ---")
    print(f"Predicted Churners: {churn_counts.get('Yes', 0):,} ({churn_counts.get('Yes', 0)/len(df_scored)*100:.1f}%)")
    print(f"Predicted Retained: {churn_counts.get('No', 0):,} ({churn_counts.get('No', 0)/len(df_scored)*100:.1f}%)")
    print("Risk Tier Distribution:")
    for tier, count in risk_counts.items():
        print(f"  - {tier:6s}: {count:,} ({count/len(df_scored)*100:.1f}%)")


def main() -> None:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Engine CLI")
    parser.add_argument("--batch", type=str, help="Path to input CSV for batch scoring")
    parser.add_argument("--output", type=str, default="data/processed/batch_predictions.csv", help="Destination path for scored CSV")
    parser.add_argument("--demo", action="store_true", help="Run demonstration on sample customer")

    args = parser.parse_args()

    if args.batch:
        score_batch_file(Path(args.batch), Path(args.output))
    else:
        score_single_customer_sample()


if __name__ == "__main__":
    main()
