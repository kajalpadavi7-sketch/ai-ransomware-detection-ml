import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# FILE PATHS
# ============================================================

VALIDATION_FILE = "data/processed/standard_validation.csv"

MODEL_FILE = "results/models/standard/random_forest.joblib"

OUTPUT_DIR = "results/evaluation/standard"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "standard_threshold_analysis.csv"
)

OUTPUT_PLOT = os.path.join(
    OUTPUT_DIR,
    "standard_threshold_analysis.png"
)

OUTPUT_THRESHOLD = os.path.join(
    OUTPUT_DIR,
    "selected_threshold.txt"
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

TARGET = "Label"


# ============================================================
# THRESHOLDS
# ============================================================

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("=" * 60)
print("LOADING STANDARD VALIDATION DATA")
print("=" * 60)

df = pd.read_csv(
    VALIDATION_FILE
)

print(
    "Validation shape:",
    df.shape
)

print(
    "Validation class distribution:"
)

print(
    df[TARGET].value_counts()
)


# ============================================================
# PREPARE X AND y
# ============================================================

X_val = df[FEATURES]

y_val = df[TARGET]


print(
    "\nFeatures:"
)

print(
    FEATURES
)


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("LOADING RANDOM FOREST MODEL")
print("=" * 60)

rf_model = joblib.load(
    MODEL_FILE
)

print(
    "Random Forest loaded successfully."
)


# ============================================================
# GET RANSOMWARE PROBABILITY
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING RANSOMWARE PROBABILITIES")
print("=" * 60)

probabilities = rf_model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# VALIDATION ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_val,
    probabilities
)

print(
    f"Validation ROC-AUC: {roc_auc:.4f}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

for threshold in THRESHOLDS:

    # --------------------------------------------------------
    # Convert probabilities into predictions
    # --------------------------------------------------------

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        y_pred
    ).ravel()

    # --------------------------------------------------------
    # False Positive Rate
    # --------------------------------------------------------

    if (tn + fp) > 0:

        fpr = fp / (tn + fp)

    else:

        fpr = 0.0

    # --------------------------------------------------------
    # False Negative Rate
    # --------------------------------------------------------

    if (tp + fn) > 0:

        fnr = fn / (tp + fn)

    else:

        fnr = 0.0

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({

        "Threshold": threshold,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": roc_auc,

        "FPR": fpr,

        "FNR": fnr,

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "TP": tp

    })

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        f"FPR      : {fpr:.4f}"
    )

    print(
        f"FNR      : {fnr:.4f}"
    )

    print(
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SELECT BEST THRESHOLD
# ============================================================
#
# Selection criterion:
# Highest F1 score on validation data.
#
# IMPORTANT:
# standard_test.csv is NOT used here.
# ============================================================

best_row = results_df.loc[
    results_df["F1"].idxmax()
]

best_threshold = float(
    best_row["Threshold"]
)

best_f1 = float(
    best_row["F1"]
)


# ============================================================
# PRINT BEST THRESHOLD
# ============================================================

print("\n" + "=" * 60)
print("BEST VALIDATION THRESHOLD")
print("=" * 60)

print(
    f"Selected threshold: {best_threshold:.2f}"
)

print(
    f"Validation F1: {best_f1:.4f}"
)

print(
    "Selection criterion: Highest Validation F1"
)


# ============================================================
# RECALL >= 90% THRESHOLDS
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLDS WITH RECALL >= 90%")
print("=" * 60)

recall_options = results_df[
    results_df["Recall"] >= 0.90
]

if len(recall_options) > 0:

    print(
        recall_options[
            [
                "Threshold",
                "Precision",
                "Recall",
                "F1",
                "FPR",
                "FNR"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No threshold achieved recall >= 90%."
    )


# ============================================================
# SAVE CSV
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    f"\nSaved threshold results:"
)

print(
    OUTPUT_CSV
)


# ============================================================
# SAVE SELECTED THRESHOLD
# ============================================================

with open(
    OUTPUT_THRESHOLD,
    "w"
) as f:

    f.write(
        f"Selected Threshold: {best_threshold:.2f}\n"
    )

    f.write(
        "Selection Criterion: Highest Validation F1\n"
    )

    f.write(
        f"Validation F1: {best_f1:.4f}\n"
    )

    f.write(
        "Test set was NOT used for threshold selection.\n"
    )


print(
    f"Saved selected threshold:"
)

print(
    OUTPUT_THRESHOLD
)


# ============================================================
# PLOT THRESHOLD VS METRICS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    results_df["Threshold"],
    results_df["Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["F1"],
    marker="o",
    label="F1"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Selected Threshold = {best_threshold:.2f}"
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Random Forest Threshold Analysis on Validation Set"
)

plt.xticks(
    THRESHOLDS
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=300
)

plt.close()


print(
    f"Saved threshold plot:"
)

print(
    OUTPUT_PLOT
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("STANDARD THRESHOLD ANALYSIS COMPLETED")
print("=" * 60)

print(
    f"Selected Threshold = {best_threshold:.2f}"
)

print(
    f"Validation F1 = {best_f1:.4f}"
)

print(
    "\nIMPORTANT:"
)

print(
    "The selected threshold will now be applied"
)

print(
    "to standard_test.csv for final evaluation."
)

print(
    "The test set was NOT used for threshold selection."
)