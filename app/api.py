"""FastAPI Application for Customer Churn Prediction Engine.

Exposes RESTful endpoints for real-time and batch customer churn scoring,
health checks, model metadata inspection, and operational metrics.
"""

import time
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from src.config import settings
from services.prediction_service import prediction_service
from app.schemas import (
    CustomerInputSchema,
    PredictionResponseSchema,
    BatchPredictionRequestSchema,
    BatchPredictionResponseSchema,
    HealthResponseSchema,
    ModelInfoResponseSchema,
    ServiceMetricsResponseSchema,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("churn_api")

# State trackers for /metrics endpoint
START_TIME = time.time()
METRICS_TRACKER = {
    "single_requests": 0,
    "batch_requests": 0,
    "records_scored": 0,
    "total_latency_seconds": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown lifecycle events."""
    logger.info("Initializing Customer Churn Prediction Engine API...")
    if prediction_service.pipeline is None:
        logger.error("Model pipeline could not be loaded on startup!")
    else:
        logger.info(f"Model successfully loaded: {prediction_service.metadata.get('model_name', 'Tuned XGBoost')}")
    yield
    logger.info("Shutting down Customer Churn Prediction Engine API.")


# Initialize FastAPI instance
app = FastAPI(
    title="Customer Churn Prediction Engine API",
    description=(
        "Production-ready REST microservice for customer churn probability scoring, "
        "calibrated risk tier categorization, and transparent feature attribution."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponseSchema,
    tags=["System"],
    summary="Check service health and model availability",
)
def get_health() -> HealthResponseSchema:
    """Return runtime service health, environment status, and model availability."""
    is_loaded = prediction_service.pipeline is not None
    return HealthResponseSchema(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        environment=settings.ENVIRONMENT,
        version=settings.VERSION,
        timestamp=datetime.now().isoformat(),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponseSchema,
    tags=["Model"],
    summary="Retrieve active model metadata and verified test metrics",
)
def get_model_info() -> ModelInfoResponseSchema:
    """Return model architecture details, optimal decision threshold, and evaluation metrics."""
    meta = prediction_service.metadata
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata is not available. Please ensure model training has been completed.",
        )

    return ModelInfoResponseSchema(
        model_name=meta.get("model_name", "Tuned XGBoost"),
        version=settings.VERSION,
        optimal_decision_threshold=meta.get("optimal_decision_threshold", settings.DEFAULT_DECISION_THRESHOLD),
        test_metrics=meta.get("test_evaluation", {}),
        pipeline_steps=meta.get("pipeline_steps", ["feature_engineer", "preprocessor", "classifier"]),
    )


@app.post(
    "/predict",
    response_model=PredictionResponseSchema,
    tags=["Inference"],
    summary="Predict churn probability for a single customer",
)
def predict_single(customer: CustomerInputSchema) -> PredictionResponseSchema:
    """Accept validated customer attributes, evaluate through pipeline, and return prediction payload."""
    t0 = time.time()
    try:
        customer_dict = customer.model_dump()
        result = prediction_service.predict_single(customer_dict)

        # Update metrics tracker
        latency = time.time() - t0
        METRICS_TRACKER["single_requests"] += 1
        METRICS_TRACKER["records_scored"] += 1
        METRICS_TRACKER["total_latency_seconds"] += latency

        logger.info(
            f"Scored customer: {result['customer_id']} | "
            f"Probability: {result['churn_probability']:.4f} | "
            f"Risk: {result['risk_level']} | Latency: {latency*1000:.2f}ms"
        )
        return PredictionResponseSchema(**result)

    except Exception as exc:
        logger.error(f"Inference error on customer {customer.customerID}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(exc)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponseSchema,
    tags=["Inference"],
    summary="Score a batch of customer records",
)
def predict_batch(batch_request: BatchPredictionRequestSchema) -> BatchPredictionResponseSchema:
    """Accept a batch of validated customer records and return individual predictions plus aggregate counts."""
    t0 = time.time()
    try:
        records = [c.model_dump() for c in batch_request.customers]
        df_batch = pd.DataFrame(records)

        # Execute batch scoring
        df_scored = prediction_service.predict_batch(
            df_batch,
            threshold=batch_request.custom_threshold,
        )

        predictions_list = []
        for _, row in df_scored.iterrows():
            cust_dict = row.to_dict()
            risk_factors = prediction_service.identify_customer_risk_factors(cust_dict)
            predictions_list.append(
                PredictionResponseSchema(
                    customer_id=str(row["customerID"]),
                    churn_probability=float(row["churn_probability"]),
                    prediction=str(row["churn_prediction"]),
                    risk_level=str(row["risk_level"]),
                    decision_threshold_applied=batch_request.custom_threshold or prediction_service.decision_threshold,
                    model_version=prediction_service.metadata.get("model_name", "Tuned XGBoost"),
                    key_risk_factors=risk_factors,
                )
            )

        churn_count = int((df_scored["churn_prediction"] == "Yes").sum())
        retain_count = int((df_scored["churn_prediction"] == "No").sum())
        high_risk_cnt = int((df_scored["risk_level"] == "High").sum())

        latency = time.time() - t0
        METRICS_TRACKER["batch_requests"] += 1
        METRICS_TRACKER["records_scored"] += len(records)
        METRICS_TRACKER["total_latency_seconds"] += latency

        logger.info(
            f"Batch inference complete: {len(records)} records scored in {latency*1000:.2f}ms | "
            f"Churners: {churn_count} | Retained: {retain_count}"
        )

        return BatchPredictionResponseSchema(
            total_records=len(records),
            total_churners_predicted=churn_count,
            total_retained_predicted=retain_count,
            high_risk_count=high_risk_cnt,
            predictions=predictions_list,
        )

    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(exc)}",
        )


@app.get(
    "/metrics",
    response_model=ServiceMetricsResponseSchema,
    tags=["System"],
    summary="Get aggregate operational and scoring metrics",
)
def get_service_metrics() -> ServiceMetricsResponseSchema:
    """Return genuine in-memory operational metrics tracked since service startup."""
    uptime = time.time() - START_TIME
    total_calls = METRICS_TRACKER["single_requests"] + METRICS_TRACKER["batch_requests"]
    avg_latency = (
        (METRICS_TRACKER["total_latency_seconds"] / total_calls * 1000.0)
        if total_calls > 0
        else 0.0
    )

    return ServiceMetricsResponseSchema(
        service_uptime_seconds=round(uptime, 2),
        total_single_predictions_served=METRICS_TRACKER["single_requests"],
        total_batch_predictions_served=METRICS_TRACKER["batch_requests"],
        total_records_scored=METRICS_TRACKER["records_scored"],
        average_latency_ms=round(avg_latency, 2),
    )
