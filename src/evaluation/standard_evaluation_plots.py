import pandas as pd
import numpy as np
import os
import joblib

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)


# ============================================================
# PATHS
# ============================================================

TEST_FILE = "data/processed/standard_test.csv"

MODEL_FILE = "results/models/standard/random_forest.joblib"

OUTPUT_DIR = "results/evaluation/standard"

ROC_CSV = os.path.join(
    OUTPUT_DIR,
    "standard_roc_curve.csv"
)

PR_CSV = os.path.join(
    OUTPUT_DIR,
    "standard_precision_recall_curve.csv"
)

ROC_PLOT = os.path.join(
    OUTPUT_DIR,
    "standard_roc_curve.png"
)

PR_PLOT = os.path.join(
    OUTPUT_DIR,
    "standard_precision_recall_curve.png"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "API_Count",
    "DLL_Count",
    "Unique_API_Count",
    "Unique_DLL_Count",
    "Size",
    "Packed",
    "Entropy",
    "Year"
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 60)
print("LOADING STANDARD TEST DATA")
print("=" * 60)

df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print("Test shape:", df.shape)


X_test = df[FEATURES]
y_test = df["Label"].astype(int)


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("LOADING RANDOM FOREST MODEL")
print("=" * 60)

model = joblib.load(
    MODEL_FILE
)

print("Random Forest loaded successfully.")


# ============================================================
# PREDICTION PROBABILITIES
# ============================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTION PROBABILITIES")
print("=" * 60)

y_prob = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# ROC CURVE
# ============================================================

print("\n" + "=" * 60)
print("GENERATING ROC CURVE")
print("=" * 60)

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)


roc_df = pd.DataFrame({
    "FPR": fpr,
    "TPR": tpr,
    "Threshold": roc_thresholds
})

roc_df.to_csv(
    ROC_CSV,
    index=False
)


# ============================================================
# PLOT ROC
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Random Forest on Standard Test Set"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    ROC_PLOT,
    dpi=300
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

print("\n" + "=" * 60)
print("GENERATING PRECISION-RECALL CURVE")
print("=" * 60)

precision, recall, pr_thresholds = precision_recall_curve(
    y_test,
    y_prob
)

average_precision = average_precision_score(
    y_test,
    y_prob
)


# ============================================================
# SAVE PR DATA
# ============================================================

pr_df = pd.DataFrame({
    "Precision": precision,
    "Recall": recall
})

pr_df.to_csv(
    PR_CSV,
    index=False
)


# ============================================================
# PLOT PR CURVE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall,
    precision,
    label=f"Random Forest (AP = {average_precision:.4f})"
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve - Random Forest on Standard Test Set"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PR_PLOT,
    dpi=300
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("STANDARD EVALUATION PLOTS COMPLETED")
print("=" * 60)

print("\nROC-AUC:", round(roc_auc, 4))

print(
    "Average Precision:",
    round(average_precision, 4)
)

print("\nSaved files:")

print(ROC_CSV)

print(PR_CSV)

print(ROC_PLOT)

print(PR_PLOT)