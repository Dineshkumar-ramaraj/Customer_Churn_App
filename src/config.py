"""Project configuration module.

Handles central directory paths, dataset URLs, model hyperparameters,
and runtime settings using Pydantic Settings and dotenv.
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory resolves to the project root (Customer_Churn_App)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application and Machine Learning settings with environment variable support."""

    # Project Information
    PROJECT_NAME: str = "Customer Churn Prediction Engine"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Dataset Source & Paths
    DATASET_URL: str = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    )
    RAW_DATA_PATH: Path = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
    PROCESSED_DATA_PATH: Path = BASE_DIR / "data" / "processed" / "customer_churn_clean.csv"

    # Artifacts & Outputs
    MODELS_DIR: Path = BASE_DIR / "models"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    FIGURES_DIR: Path = BASE_DIR / "reports" / "figures"
    EVALUATION_DIR: Path = BASE_DIR / "reports" / "evaluation"
    INSIGHTS_DIR: Path = BASE_DIR / "reports" / "insights"

    # Modeling Parameters
    TARGET_COLUMN: str = "Churn"
    ID_COLUMN: str = "customerID"
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.20
    DEFAULT_DECISION_THRESHOLD: float = 0.50

    # Expected raw column schema for validation
    EXPECTED_COLUMNS: List[str] = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    # Server Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    STREAMLIT_PORT: int = 8501

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Global settings singleton
settings = Settings()
