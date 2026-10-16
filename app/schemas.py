"""Pydantic v2 schemas for Customer Churn API request validation and response models."""

from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class CustomerInputSchema(BaseModel):
    """Schema validating individual customer attributes for churn prediction."""

    customerID: Optional[str] = Field(default="CUST-0000", description="Unique customer identifier")
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether the customer is a senior citizen (1, 0)")
    Partner: Literal["Yes", "No"] = Field(..., description="Whether the customer has a partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Whether the customer has dependents")
    tenure: int = Field(..., ge=0, le=120, description="Number of months customer has stayed with company")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Whether customer has phone service")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(..., description="Whether customer has multiple lines")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Customer's internet service provider")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(..., description="Online security add-on status")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., description="Online backup add-on status")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., description="Device protection add-on status")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(..., description="Tech support add-on status")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(..., description="Streaming TV add-on status")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(..., description="Streaming movies add-on status")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contract term commitment")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Whether customer has paperless billing")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Payment method used by the customer")
    MonthlyCharges: float = Field(..., ge=0.0, le=500.0, description="Monthly recurring charge amount in USD")
    TotalCharges: Optional[Union[float, str]] = Field(default=0.0, description="Total cumulative charge amount in USD")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    )


class PredictionResponseSchema(BaseModel):
    """Schema for individual customer prediction response."""

    customer_id: str = Field(..., description="Identifier of the scored customer")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of churning")
    prediction: Literal["Yes", "No"] = Field(..., description="Binary churn classification")
    risk_level: Literal["Low", "Medium", "High"] = Field(..., description="Configurable risk category")
    decision_threshold_applied: float = Field(..., description="Classification threshold used for prediction")
    model_version: str = Field(..., description="Version of the active model pipeline")
    key_risk_factors: List[str] = Field(..., description="Primary risk drivers explaining this prediction")


class BatchPredictionRequestSchema(BaseModel):
    """Schema for batch prediction requests."""

    customers: List[CustomerInputSchema] = Field(..., min_length=1, max_length=5000)
    custom_threshold: Optional[float] = Field(default=None, ge=0.01, le=0.99)


class BatchPredictionResponseSchema(BaseModel):
    """Schema for batch prediction response."""

    total_records: int
    total_churners_predicted: int
    total_retained_predicted: int
    high_risk_count: int
    predictions: List[PredictionResponseSchema]


class HealthResponseSchema(BaseModel):
    """Schema for service health check response."""

    status: str
    model_loaded: bool
    environment: str
    version: str
    timestamp: str


class ModelInfoResponseSchema(BaseModel):
    """Schema for active model information and verified evaluation metrics."""

    model_name: str
    version: str
    optimal_decision_threshold: float
    test_metrics: Dict[str, Any]
    pipeline_steps: List[str]


class ServiceMetricsResponseSchema(BaseModel):
    """Schema for safe, aggregate service operational metrics."""

    service_uptime_seconds: float
    total_single_predictions_served: int
    total_batch_predictions_served: int
    total_records_scored: int
    average_latency_ms: float
