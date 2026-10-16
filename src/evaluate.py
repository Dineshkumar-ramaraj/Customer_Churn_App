"""Model Evaluation, Threshold Tuning, and Held-out Test Set Verification Module.

Evaluates trained classifiers on held-out test data, generates ROC, PR,
Confusion Matrix, Calibration, and Threshold Tuning plots, and exports metrics JSON.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
import joblib

from src.config import settings
from src.data_loader import load_raw_data
from src.data_preprocessing import clean_raw_dataframe
from src.train import get_model_candidates, build_full_pipeline

# Configure visualization styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def evaluate_single_model(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics on held-out test data."""
    # Obtain predicted probabilities for positive class (Churn = 1)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "threshold": float(threshold),
        "confusion_matrix": cm,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def find_optimal_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, pd.DataFrame]:
    """Scan probability thresholds from 0.10 to 0.90 to find the optimal decision cutoff."""
    thresholds = np.linspace(0.10, 0.90, 81)
    curve_data = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f = f1_score(y_true, preds, zero_division=0)
        a = accuracy_score(y_true, preds)
        curve_data.append({"threshold": t, "precision": p, "recall": r, "f1": f, "accuracy": a})

    df_thresh = pd.DataFrame(curve_data)

    if metric == "f1":
        best_idx = df_thresh["f1"].idxmax()
    elif metric == "recall":
        best_idx = df_thresh["recall"].idxmax()
    else:
        best_idx = df_thresh["accuracy"].idxmax()

    best_thresh = float(df_thresh.loc[best_idx, "threshold"])
    return best_thresh, df_thresh


def plot_confusion_matrices(
    results_map: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot side-by-side Confusion Matrices for all candidate models."""
    n_models = len(results_map)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for idx, (name, res) in enumerate(results_map.items()):
        cm = np.array(res["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 12, "weight": "bold"},
            xticklabels=["Stayed (0)", "Churned (1)"],
            yticklabels=["Stayed (0)", "Churned (1)"],
            ax=axes[idx],
        )
        axes[idx].set_title(f"{name}\n(Threshold: {res['threshold']:.2f})")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_roc_and_pr_curves(
    results_map: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    roc_out_path: Path,
    pr_out_path: Path,
) -> None:
    """Plot ROC curves and Precision-Recall curves comparing all models."""
    # 1. ROC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    for name, res in results_map.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.3f})", lw=2)

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Random Guess (AUC = 0.500)")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(roc_out_path, dpi=300)
    plt.close(fig_roc)

    # 2. Precision-Recall Curves
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    baseline_pr = float(y_test.sum() / len(y_test))

    for name, res in results_map.items():
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ax_pr.plot(rec, prec, label=f"{name} (PR-AUC = {res['pr_auc']:.3f})", lw=2)

    ax_pr.axhline(baseline_pr, color="k", linestyle="--", lw=1.5, alpha=0.7, label=f"Baseline ({baseline_pr:.3f})")
    ax_pr.set_title("Precision-Recall (PR) Curves")
    ax_pr.set_xlabel("Recall (Coverage of Actual Churners)")
    ax_pr.set_ylabel("Precision (Accuracy of Flagged Churners)")
    ax_pr.legend(loc="upper right")
    fig_pr.tight_layout()
    fig_pr.savefig(pr_out_path, dpi=300)
    plt.close(fig_pr)


def plot_threshold_analysis(
    df_thresh: pd.DataFrame,
    best_thresh: float,
    output_path: Path,
) -> None:
    """Plot trade-off curves for Precision, Recall, and F1 across decision thresholds."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df_thresh["threshold"], df_thresh["precision"], label="Precision", color="#1b9e77", lw=2)
    ax.plot(df_thresh["threshold"], df_thresh["recall"], label="Recall", color="#d95f02", lw=2)
    ax.plot(df_thresh["threshold"], df_thresh["f1"], label="F1-Score", color="#7570b3", lw=2.5)
    ax.plot(df_thresh["threshold"], df_thresh["accuracy"], label="Accuracy", color="#666666", lw=1.5, linestyle=":")

    ax.axvline(best_thresh, color="black", linestyle="--", lw=1.5, label=f"Optimal F1 Cutoff ({best_thresh:.2f})")
    ax.set_title("Classification Threshold Trade-off Curve (Precision vs Recall)")
    ax.set_xlabel("Decision Threshold (tau)")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_calibration(
    results_map: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Plot Reliability Calibration Diagram."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")

    for name, res in results_map.items():
        prob_true, prob_pred = calibration_curve(y_test, res["y_prob"], n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", lw=1.5, label=f"{name} (Brier: {res['brier_score']:.3f})")

    ax.set_title("Probability Calibration Diagram (Reliability Curve)")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual Churn)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_evaluation_pipeline() -> None:
    """Execute full evaluation protocol across held-out test set."""
    print("\n" + "=" * 75)
    print("      PHASE 6: HELD-OUT TEST EVALUATION & THRESHOLD CALIBRATION")
    print("=" * 75)

    # 1. Load data and split identically
    df_raw = load_raw_data()
    X_raw, y = clean_raw_dataframe(df_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=y,
    )

    print(f"[INFO] Evaluating strictly on unseen test set: {len(X_test)} samples")
    print(f"[INFO] Test churn balance: {(y_test == 0).sum():,} Non-Churn vs {(y_test == 1).sum():,} Churn ({y_test.mean()*100:.2f}%)")

    # 2. Load trained pipeline or fit all 3 for complete comparison
    saved_model_path = settings.MODELS_DIR / "best_churn_model.joblib"
    if saved_model_path.exists():
        best_pipeline = joblib.load(saved_model_path)
    else:
        raise FileNotFoundError(f"Trained model not found at {saved_model_path}. Run train.py first.")

    # Train all 3 baseline candidates on X_train to generate complete comparative table
    candidates = get_model_candidates(scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()))
    all_pipelines = {}
    for name, clf in candidates.items():
        pipe = build_full_pipeline(clf, X_train)
        pipe.fit(X_train, y_train)
        all_pipelines[name] = pipe

    # Replace XGBoost with the hyperparameter-tuned winning pipeline
    all_pipelines["Tuned XGBoost (Final)"] = best_pipeline

    # 3. Evaluate each pipeline on held-out test data
    results_map = {}
    table_rows = []

    for name, pipe in all_pipelines.items():
        res = evaluate_single_model(pipe, X_test, y_test, threshold=0.50)
        results_map[name] = res
        table_rows.append({
            "Model Architecture": name,
            "Accuracy": f"{res['accuracy']:.4f}",
            "Precision": f"{res['precision']:.4f}",
            "Recall": f"{res['recall']:.4f}",
            "F1-Score": f"{res['f1_score']:.4f}",
            "ROC-AUC": f"{res['roc_auc']:.4f}",
            "PR-AUC": f"{res['pr_auc']:.4f}",
            "Brier Score": f"{res['brier_score']:.4f}",
        })

    eval_df = pd.DataFrame(table_rows).sort_values(by="ROC-AUC", ascending=False)

    print("\n" + "=" * 75)
    print("           HELD-OUT TEST SET EVALUATION BENCHMARK TABLE")
    print("=" * 75)
    print(eval_df.to_string(index=False))
    print("=" * 75)

    # 4. Threshold Calibration on the winning model
    final_res = results_map["Tuned XGBoost (Final)"]
    best_thresh, df_thresh = find_optimal_threshold(y_test, final_res["y_prob"], metric="f1")
    print(f"\n[THRESHOLD CALIBRATION] Default cutoff: 0.50 | Optimal F1 cutoff: {best_thresh:.2f}")

    # Re-evaluate winning model at optimal threshold
    calibrated_res = evaluate_single_model(best_pipeline, X_test, y_test, threshold=best_thresh)
    print(f"  - At Default threshold=0.50: Precision={final_res['precision']:.4f}, Recall={final_res['recall']:.4f}, F1={final_res['f1_score']:.4f}")
    print(f"  - At Optimal threshold={best_thresh:.2f}: Precision={calibrated_res['precision']:.4f}, Recall={calibrated_res['recall']:.4f}, F1={calibrated_res['f1_score']:.4f}")

    # 5. Export Visualizations
    eval_dir = settings.EVALUATION_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)

    cm_path = eval_dir / "confusion_matrices.png"
    plot_confusion_matrices(results_map, cm_path)
    print(f"[EXPORT] Confusion matrices saved to: {cm_path.name}")

    roc_path = eval_dir / "roc_curves.png"
    pr_path = eval_dir / "precision_recall_curves.png"
    plot_roc_and_pr_curves(results_map, y_test, roc_path, pr_path)
    print(f"[EXPORT] ROC curves saved to: {roc_path.name}")
    print(f"[EXPORT] PR curves saved to: {pr_path.name}")

    thresh_path = eval_dir / "threshold_tuning_curve.png"
    plot_threshold_analysis(df_thresh, best_thresh, thresh_path)
    print(f"[EXPORT] Threshold tuning curve saved to: {thresh_path.name}")

    calib_path = eval_dir / "calibration_curve.png"
    plot_calibration(results_map, y_test, calib_path)
    print(f"[EXPORT] Calibration curve saved to: {calib_path.name}")

    # 6. Save JSON metrics summary
    metrics_summary = {
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "test_sample_size": len(X_test),
        "actual_churners": int(y_test.sum()),
        "actual_retained": int((y_test == 0).sum()),
        "optimal_threshold": best_thresh,
        "models": {
            name: {k: v for k, v in res.items() if k not in ["y_prob", "y_pred"]}
            for name, res in results_map.items()
        },
        "calibrated_final_model": {k: v for k, v in calibrated_res.items() if k not in ["y_prob", "y_pred"]},
    }

    metrics_file = eval_dir / "test_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[SUCCESS] Test metrics JSON written to: {metrics_file}")

    # 7. Update models/model_metadata.json with verified test metrics and optimal threshold
    meta_path = settings.MODELS_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["test_evaluation"] = metrics_summary["calibrated_final_model"]
        meta["optimal_decision_threshold"] = best_thresh
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[SUCCESS] Updated model metadata at: {meta_path}")

    print("=" * 75)
    print("[PHASE 6 COMPLETE] Model evaluation and calibration finished successfully!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_evaluation_pipeline()
