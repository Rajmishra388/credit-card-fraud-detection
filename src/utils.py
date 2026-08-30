"""
Shared utilities for model evaluation, metric logging, and plot saving.
Used by every model training script to keep results consistent and comparable.
"""
import matplotlib
matplotlib.use("Agg")
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve
)

METRICS_FILE = "results/metrics.csv"
RESULTS_DIR = "results"


def evaluate_and_log(model_name, y_true, y_pred, y_proba):
    """
    Computes standard classification metrics, saves confusion matrix and
    ROC curve plots, and appends a row to the shared results/metrics.csv.
    """
    os.makedirs(f"{RESULTS_DIR}/{model_name}", exist_ok=True)

    # ---- Compute metrics ----
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    # ---- Confusion matrix plot ----
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}/confusion_matrix.png", dpi=150)
    plt.close()

    # ---- ROC curve plot ----
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}/roc_curve.png", dpi=150)
    plt.close()

    # ---- Append to shared metrics CSV ----
    row = pd.DataFrame([{
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }])

        os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(METRICS_FILE):
        existing = pd.read_csv(METRICS_FILE)
        existing = existing[existing["model"] != model_name]  # drop old row for this model, if any
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(METRICS_FILE, mode="w", header=True, index=False)
    print(f"Logged results to {METRICS_FILE}")
    print(f"Plots saved to {RESULTS_DIR}/{model_name}/")


def load_processed_data():
    """Loads the preprocessed train/test splits saved by preprocess.py."""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test