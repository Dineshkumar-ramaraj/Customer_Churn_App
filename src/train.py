"""Model Training and Benchmarking Module for Customer Churn Engine.

Trains, cross-validates, and compares Logistic Regression, Random Forest,
and XGBoost classifiers using reproducible Scikit-Learn pipelines.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
import joblib

from src.config import settings
from src.data_loader import load_raw_data
from src.data_preprocessing import clean_raw_dataframe, identify_feature_types, build_preprocessor
from src.feature_engineering import CustomerFeatureEngineer


def get_model_candidates(scale_pos_weight: float = 2.77) -> Dict[str, Any]:
    """Define candidate classification models with balanced class handling.

    Args:
        scale_pos_weight: Ratio of negative to positive classes (for XGBoost).

    Returns:
        Dictionary of initialized model instances.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=settings.RANDOM_STATE,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=settings.RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=settings.RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def build_full_pipeline(classifier: Any, X_sample: pd.DataFrame) -> Pipeline:
    """Construct an end-to-end Scikit-Learn pipeline chaining feature engineering,
    preprocessing (ColumnTransformer), and the classifier.

    Args:
        classifier: Estimator instance (LogisticRegression, RandomForest, or XGBoost).
        X_sample: Sample raw feature DataFrame to infer feature names and types.

    Returns:
        Complete end-to-end Pipeline.
    """
    fe = CustomerFeatureEngineer()
    X_engineered_sample = fe.fit_transform(X_sample)

    num_cols, cat_cols = identify_feature_types(X_engineered_sample)
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=True)

    pipeline = Pipeline(
        steps=[
            ("feature_engineer", fe),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def benchmark_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """Run Stratified K-Fold Cross-Validation across all candidate models.

    Args:
        X_train: Training features DataFrame (raw attributes).
        y_train: Training target binary Series.
        cv_splits: Number of stratified folds.

    Returns:
        Tuple of (results_summary_df, fitted_pipelines_dict).
    """
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_weight = float(neg_count / max(1, pos_count))

    models = get_model_candidates(scale_pos_weight=scale_weight)
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=settings.RANDOM_STATE)

    scoring = {
        "roc_auc": "roc_auc",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "accuracy": "accuracy",
    }

    results = []
    fitted_pipelines = {}

    print(f"\n[INFO] Starting Stratified {cv_splits}-Fold Cross-Validation on Training Split ({len(X_train)} samples)...")
    print(f"[INFO] Class balance: {neg_count:,} negative (0) vs {pos_count:,} positive (1) | Scale Weight: {scale_weight:.2f}")
    print("-" * 75)

    for name, clf in models.items():
        print(f"--> Training & Cross-Validating: {name}...")
        pipeline = build_full_pipeline(clf, X_train)

        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        mean_roc = cv_scores["test_roc_auc"].mean()
        std_roc = cv_scores["test_roc_auc"].std()
        mean_recall = cv_scores["test_recall"].mean()
        mean_precision = cv_scores["test_precision"].mean()
        mean_f1 = cv_scores["test_f1"].mean()
        mean_acc = cv_scores["test_accuracy"].mean()

        results.append({
            "Model": name,
            "CV ROC-AUC": f"{mean_roc:.4f} (+/- {std_roc:.4f})",
            "CV Recall": f"{mean_recall:.4f}",
            "CV Precision": f"{mean_precision:.4f}",
            "CV F1-Score": f"{mean_f1:.4f}",
            "CV Accuracy": f"{mean_acc:.4f}",
            "_raw_roc": mean_roc,
            "_raw_f1": mean_f1,
            "_raw_recall": mean_recall,
        })

        # Fit full pipeline on all training data
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

    summary_df = pd.DataFrame(results).sort_values(by="_raw_roc", ascending=False)
    return summary_df, fitted_pipelines


def tune_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_model_name: str,
) -> Pipeline:
    """Perform targeted grid search hyperparameter tuning on the leading model candidate."""
    print(f"\n[INFO] Tuning hyperparameters for top model: {best_model_name}...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.RANDOM_STATE)

    if best_model_name == "XGBoost":
        base_clf = XGBClassifier(
            scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
            random_state=settings.RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        )
        pipeline = build_full_pipeline(base_clf, X_train)
        param_grid = {
            "classifier__max_depth": [3, 4, 5],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__n_estimators": [100, 150, 200],
            "classifier__subsample": [0.8, 1.0],
        }
    elif best_model_name == "Logistic Regression":
        base_clf = LogisticRegression(
            class_weight="balanced",
            random_state=settings.RANDOM_STATE,
            max_iter=1000,
        )
        pipeline = build_full_pipeline(base_clf, X_train)
        param_grid = {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],
        }
    else:  # Random Forest
        base_clf = RandomForestClassifier(
            class_weight="balanced",
            random_state=settings.RANDOM_STATE,
            n_jobs=-1,
        )
        pipeline = build_full_pipeline(base_clf, X_train)
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [8, 12, 16],
            "classifier__min_samples_split": [2, 5],
        }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print(f"[SUCCESS] Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"[SUCCESS] Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def save_trained_model(
    pipeline: Pipeline,
    model_name: str,
    benchmark_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save the winning fitted pipeline and training metadata JSON."""
    out_dir = output_dir or settings.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "best_churn_model.joblib"
    joblib.dump(pipeline, model_path)

    # Save model metadata
    metadata = {
        "model_name": model_name,
        "saved_at": datetime.now().isoformat(),
        "random_state": settings.RANDOM_STATE,
        "test_size": settings.TEST_SIZE,
        "benchmark_summary": benchmark_df.drop(columns=["_raw_roc", "_raw_f1", "_raw_recall"]).to_dict(orient="records"),
        "pipeline_steps": [name for name, _ in pipeline.steps],
    }

    meta_path = out_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SUCCESS] Serialized winning model pipeline to: {model_path}")
    print(f"[SUCCESS] Model metadata logged to: {meta_path}")
    return model_path


def main() -> None:
    """Execute complete Phase 5 model training and benchmarking workflow."""
    print("\n" + "=" * 75)
    print("      PHASE 5: MODEL TRAINING, BENCHMARKING & PIPELINE SELECTION")
    print("=" * 75)

    # 1. Load data and split using stratified train_test_split
    from sklearn.model_selection import train_test_split
    df_raw = load_raw_data()
    X_raw, y = clean_raw_dataframe(df_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=y,
    )

    # 2. Benchmark candidate models with 5-fold CV
    summary_df, fitted_pipelines = benchmark_models(X_train, y_train, cv_splits=5)

    print("\n" + "=" * 75)
    print("        CROSS-VALIDATION BENCHMARK RESULTS (TRAINING DATA ONLY)")
    print("=" * 75)
    display_df = summary_df.drop(columns=["_raw_roc", "_raw_f1", "_raw_recall"])
    print(display_df.to_string(index=False))
    print("=" * 75)

    # 3. Identify best model by ROC-AUC
    best_row = summary_df.iloc[0]
    best_name = best_row["Model"]
    print(f"\n[TOP PERFORMER] '{best_name}' achieved highest CV ROC-AUC: {best_row['CV ROC-AUC']}")

    # 4. Hyperparameter tuning on top model
    tuned_pipeline = tune_best_model(X_train, y_train, best_name)

    # 5. Save winning pipeline and metadata
    save_trained_model(tuned_pipeline, best_name, summary_df)

    print("\n[PHASE 5 COMPLETE] Pipeline is ready for Phase 6 evaluation on the held-out test set!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
