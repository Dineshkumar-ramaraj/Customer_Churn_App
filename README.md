# 🚀 Customer Churn Prediction Engine

An industry-grade, end-to-end Machine Learning system designed to predict customer churn, identify key retention risk drivers, and serve real-time predictions via a FastAPI backend and an interactive Streamlit dashboard.

---

## 📌 Project Overview

Customer churn occurs when customers stop doing business with a company. For subscription and recurring revenue businesses (such as telecommunications, SaaS, and utilities), acquiring a new customer is significantly more expensive than retaining an existing one.

This project delivers:
1. **Exploratory Data Analysis (EDA):** Deep domain insight into behavioral patterns driving attrition.
2. **Robust Data Pipeline:** Leakage-free preprocessing, validation, and feature engineering with Scikit-learn.
3. **Model Benchmarking:** Comparison of Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning.
4. **Model Explainability:** Global and local interpretability using feature importances and SHAP.
5. **FastAPI Inference Microservice:** Production-ready RESTful API for real-time and batch predictions.
6. **Streamlit Analytics Dashboard:** Interactive business dashboard for executives and retention managers.
7. **Containerization & CI/CD Readiness:** Dockerized deployment with automated test suites.

---

## 📂 Project Architecture

```text
Customer_Churn_App/
│
├── data/
│   ├── raw/                 # Original, immutable IBM Telco dataset
│   └── processed/           # Transformed datasets ready for modeling
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Centralized settings and paths
│   ├── data_loader.py       # Automated dataset retrieval and loader
│   ├── data_validation.py   # Schema integrity and missing value validator
│   ├── data_preprocessing.py# Leak-free sklearn pipelines
│   ├── feature_engineering.py
│   ├── train.py             # Reproducible training pipeline
│   ├── evaluate.py          # Metric calculations and threshold tuning
│   ├── predict.py           # Core inference engine
│   └── explainability.py    # Model interpretability
│
├── models/                  # Serialized pipelines and model metadata
├── reports/
│   ├── figures/             # Visualizations & charts
│   ├── evaluation/          # Confusion matrices, ROC curves, metric JSONs
│   └── insights/            # Actionable business insights
│
├── app/
│   ├── __init__.py
│   ├── api.py               # FastAPI application
│   ├── schemas.py           # Pydantic request/response schemas
│   └── dashboard.py         # Streamlit multi-page dashboard
│
├── services/
│   ├── __init__.py
│   └── prediction_service.py# Business inference service
│
├── tests/                   # Pytest automated test suite
├── requirements.txt         # Pinned project dependencies
├── .env.example             # Environment configuration template
├── .gitignore
├── Dockerfile
└── README.md
```

---

## 🛠️ Technology Stack

- **Data Processing & ML:** Python 3.13, Pandas, NumPy, Scikit-learn, XGBoost, Joblib
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Backend API:** FastAPI, Uvicorn, Pydantic v2
- **Frontend Dashboard:** Streamlit
- **Testing & Quality:** Pytest, HTTPX
- **Deployment & Config:** Docker, Python-dotenv

---

## ⚡ Quickstart: Setup and Execution

### 1. Environment Setup (Windows PowerShell)
```powershell
# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download and Verify Dataset
```powershell
python -m src.data_loader
```

### 3. Run Automated Tests
```powershell
pytest -v
```

---

*Author: Dineshkumar Ramaraj*  
*Project Status: Phase 1 (Environment Setup & Data Ingestion) Complete*
