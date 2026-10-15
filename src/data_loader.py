"""Data loading module for Customer Churn Prediction Engine.

Handles automated dataset downloading, raw file verification,
and loading into Pandas DataFrame.
"""

import sys
import urllib.request
from pathlib import Path
from typing import Optional
import pandas as pd

from src.config import settings
from src.data_validation import validate_raw_data, print_validation_report


def download_dataset(
    url: Optional[str] = None,
    output_path: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download the raw IBM Telco Customer Churn dataset if not already present.

    Args:
        url: Direct HTTP URL to raw CSV. Defaults to settings.DATASET_URL.
        output_path: Destination Path on disk. Defaults to settings.RAW_DATA_PATH.
        force: If True, redownload even if file already exists.

    Returns:
        Path to downloaded file.
    """
    url = url or settings.DATASET_URL
    output_path = output_path or settings.RAW_DATA_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        file_size_kb = output_path.stat().st_size / 1024
        print(f"[INFO] Dataset already exists at {output_path} ({file_size_kb:.2f} KB). Skipping download.")
        return output_path

    print(f"[INFO] Downloading dataset from:\n       {url}")
    print(f"[INFO] Saving to:\n       {output_path}")

    try:
        # Download with headers to prevent HTTP 403 blocks
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req) as response, open(output_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)

        file_size_kb = output_path.stat().st_size / 1024
        print(f"[SUCCESS] Dataset downloaded successfully ({file_size_kb:.2f} KB).")
        return output_path
    except Exception as exc:
        print(f"[ERROR] Failed to download dataset: {exc}", file=sys.stderr)
        raise


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw dataset from disk into a Pandas DataFrame.

    Args:
        data_path: Path to raw CSV. Defaults to settings.RAW_DATA_PATH.

    Returns:
        Pandas DataFrame of the raw dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    path = data_path or settings.RAW_DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Run download_dataset() first or specify a valid path."
        )

    df = pd.read_csv(path)
    return df


def inspect_raw_dataframe(df: pd.DataFrame) -> None:
    """Print structural inspection details of the raw DataFrame."""
    print("\n" + "=" * 60)
    print("        RAW DATASET STRUCTURAL INSPECTION")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("--- First 3 Records ---")
    print(df.head(3))

    print("\n--- Last 3 Records ---")
    print(df.tail(3))

    print("\n--- Column Data Types & Non-Null Counts ---")
    print(df.info(verbose=True, show_counts=True))

    print("\n--- Summary Statistics (Numerical Columns) ---")
    print(df.describe().T)

    print("\n--- Summary Statistics (Categorical & String Columns) ---")
    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    if len(categorical_cols) > 0:
        print(df[categorical_cols].describe().T)
    print("=" * 60 + "\n")


def main() -> None:
    """CLI execution entrypoint for Phase 1 verification."""
    print("\n[Phase 1] Starting Dataset Retrieval and Validation Pipeline...")
    # Step 1: Ensure raw dataset is present
    download_dataset()

    # Step 2: Load raw dataset
    df = load_raw_data()

    # Step 3: Run schema and integrity validation
    validation_results = validate_raw_data(df)
    print_validation_report(validation_results)

    # Step 4: Run structural inspection
    inspect_raw_dataframe(df)

    print("[Phase 1] Dataset loading, verification, and inspection completed successfully!\n")


if __name__ == "__main__":
    main()
