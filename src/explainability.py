"""Model Explainability and Interpretability Module for Customer Churn Engine.

Generates global and local feature attribution using:
1. Logistic Regression Coefficients & Odds Ratios.
2. XGBoost Feature Importance (Gain).
3. Model-Agnostic Permutation Importance on Held-out Test Set.
4. SHAP (SHapley Additive exPlanations) Global Summary and Local Waterfall Plots.
"""

from pathlib import Path
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import joblib
import shap

from src.config import settings
from src.data_loader import load_raw_data
from src.data_preprocessing import clean_raw_dataframe
from src.train import get_model_candidates, build_full_pipeline

# Styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def analyze_logistic_regression_coefficients(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    """Extract, rank, and visualize Logistic Regression Log-Odds coefficients and Odds Ratios."""
    print("[INFO] Computing Logistic Regression coefficients and Odds Ratios...")
    base_clf = get_model_candidates(scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()))["Logistic Regression"]
    pipeline = build_full_pipeline(base_clf, X_train)
    pipeline.fit(X_train, y_train)

    # Extract feature names after preprocessing
    fe = pipeline.named_steps["feature_engineer"]
    prep = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    X_eng = fe.transform(X_train)
    feature_names = list(prep.get_feature_names_out())
    coefs = clf.coef_[0]

    df_coef = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient (Log-Odds)": coefs,
        "Odds Ratio": np.exp(coefs),
    }).sort_values(by="Coefficient (Log-Odds)", ascending=False)

    # Plot top 10 positive (risk drivers) and top 10 negative (retention protectors)
    top_pos = df_coef.head(10)
    top_neg = df_coef.tail(10)
    df_plot = pd.concat([top_pos, top_neg]).sort_values(by="Coefficient (Log-Odds)", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#1b9e77" if c < 0 else "#d95f02" for c in df_plot["Coefficient (Log-Odds)"]]
    bars = ax.barh(df_plot["Feature"], df_plot["Coefficient (Log-Odds)"], color=colors, edgecolor="black", height=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w:+.3f}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(4 if w >= 0 else -4, 0),
            textcoords="offset points",
            ha="left" if w >= 0 else "right",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Logistic Regression Standardized Coefficients (Log-Odds of Churn)", pad=15)
    ax.set_xlabel("Coefficient (Beta)")
    ax.axvline(0, color="black", linestyle="--", lw=1)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return df_coef


def analyze_xgboost_feature_importance(
    pipeline: Any,
    output_path: Path,
) -> pd.DataFrame:
    """Extract and plot XGBoost built-in Feature Importance by Gain."""
    print("[INFO] Computing XGBoost split gain feature importance...")
    prep = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]
    feature_names = list(prep.get_feature_names_out())

    # Obtain feature importances (Gain)
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance (Gain)": importances,
    }).sort_values(by="Importance (Gain)", ascending=False)

    top20 = df_imp.head(20).sort_values(by="Importance (Gain)", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top20["Feature"], top20["Importance (Gain)"], color="#2b5c8f", edgecolor="black", height=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w:.4f}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Top 20 XGBoost Feature Importances (Average Gain)", pad=15)
    ax.set_xlabel("Relative Information Gain")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return df_imp


def analyze_permutation_importance(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    """Compute model-agnostic Permutation Importance on the held-out test set."""
    print("[INFO] Computing Permutation Importance on held-out test data (5 repeats)...")
    perm_res = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=settings.RANDOM_STATE,
        n_jobs=-1,
    )

    df_perm = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean_Importance": perm_res.importances_mean,
        "Std_Importance": perm_res.importances_std,
    }).sort_values(by="Mean_Importance", ascending=False)

    df_plot = df_perm.head(15).sort_values(by="Mean_Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        df_plot["Feature"],
        df_plot["Mean_Importance"],
        xerr=df_plot["Std_Importance"],
        color="#7570b3",
        edgecolor="black",
        height=0.6,
        capsize=4,
    )
    ax.set_title("Model-Agnostic Permutation Importance (Drop in Test ROC-AUC)", pad=15)
    ax.set_xlabel("Mean Decrease in ROC-AUC when Feature is Shuffled")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return df_perm


def analyze_shap_explanations(
    pipeline: Any,
    X_test: pd.DataFrame,
    summary_out_path: Path,
    waterfall_out_path: Path,
) -> None:
    """Compute exact SHAP values using TreeExplainer and export Global & Local plots."""
    print("[INFO] Computing SHAP values using TreeExplainer on transformed test features...")
    fe = pipeline.named_steps["feature_engineer"]
    prep = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    # Transform test features through the preprocessing stages
    X_test_eng = fe.transform(X_test)
    feature_names = list(prep.get_feature_names_out())
    X_test_trans = prep.transform(X_test_eng)
    df_trans = pd.DataFrame(X_test_trans, columns=feature_names)

    # Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    # Sample up to 500 test instances for fast, precise SHAP distribution
    sample_df = df_trans.iloc[:500]
    shap_values = explainer(sample_df)

    # 1. Global SHAP Summary Beeswarm Plot
    fig_summary, ax_summary = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values.values,
        sample_df,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.title("SHAP Global Feature Attribution (Beeswarm Distribution)", pad=15)
    plt.tight_layout()
    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(summary_out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"[EXPORT] SHAP summary plot saved to: {summary_out_path.name}")

    # 2. Local Waterfall Plot for a Representative High-Risk Churner
    # Find sample with highest predicted probability
    probs = model.predict_proba(sample_df)[:, 1]
    high_risk_idx = int(np.argmax(probs))

    fig_waterfall, ax_wf = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[high_risk_idx], max_display=10, show=False)
    plt.title(f"SHAP Local Attribution: High-Risk Customer (Predicted Prob: {probs[high_risk_idx]:.3f})", pad=15)
    plt.tight_layout()
    plt.savefig(waterfall_out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"[EXPORT] SHAP local waterfall plot saved to: {waterfall_out_path.name}")


def run_explainability_pipeline() -> None:
    """Execute complete Phase 10 model explainability workflow."""
    print("\n" + "=" * 75)
    print("      PHASE 10: MODEL EXPLAINABILITY & INTERPRETABILITY")
    print("=" * 75)

    # 1. Load data and split
    df_raw = load_raw_data()
    X_raw, y = clean_raw_dataframe(df_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=y,
    )

    # 2. Load trained pipeline
    model_path = settings.MODELS_DIR / "best_churn_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train model first.")
    best_pipeline = joblib.load(model_path)

    fig_dir = settings.FIGURES_DIR
    insights_dir = settings.INSIGHTS_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    insights_dir.mkdir(parents=True, exist_ok=True)

    # 3. Method 1: Logistic Regression Coefficients
    coef_img = fig_dir / "08_logistic_regression_coefficients.png"
    df_coef = analyze_logistic_regression_coefficients(X_train, y_train, coef_img)

    # 4. Method 2: XGBoost Feature Importance
    xgb_img = fig_dir / "09_xgboost_feature_importance.png"
    df_xgb = analyze_xgboost_feature_importance(best_pipeline, xgb_img)

    # 5. Method 3: Permutation Importance
    perm_img = fig_dir / "10_permutation_feature_importance.png"
    df_perm = analyze_permutation_importance(best_pipeline, X_test, y_test, perm_img)

    # 6. Method 4: SHAP Explanations
    shap_summary_img = fig_dir / "11_shap_summary_beeswarm.png"
    shap_wf_img = fig_dir / "12_shap_individual_waterfall.png"
    analyze_shap_explanations(best_pipeline, X_test, shap_summary_img, shap_wf_img)

    # 7. Persist JSON Summary
    summary_data = {
        "top_xgboost_gain_features": df_xgb.head(10).to_dict(orient="records"),
        "top_permutation_features": df_perm.head(10).to_dict(orient="records"),
        "top_positive_log_odds_features": df_coef.head(5).to_dict(orient="records"),
        "top_negative_log_odds_features": df_coef.tail(5).to_dict(orient="records"),
    }
    json_path = insights_dir / "feature_importance_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[SUCCESS] Feature importance summary logged to: {json_path}")

    print("=" * 75)
    print("[PHASE 10 COMPLETE] Model explainability pipeline executed successfully!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_explainability_pipeline()
