import os
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = "data/processed/test_unseen_family.csv"
MODEL_FILE = "results/models/random_forest.joblib"

OUTPUT_DIR = "results/evaluation"

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

# Thresholds to evaluate
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

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 50)
print("LOADING TEST DATA")
print("=" * 50)

df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print("Test shape:", df.shape)


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

print("\n" + "=" * 50)
print("LOADING RANDOM FOREST MODEL")
print("=" * 50)

model = joblib.load(MODEL_FILE)

print("Random Forest loaded successfully.")


# ============================================================
# PREPARE DATA
# ============================================================

X_test = df[FEATURES]
y_test = df["Label"]


# ============================================================
# GET RANSOMWARE PROBABILITY
# ============================================================

print("\n" + "=" * 50)
print("CALCULATING PREDICTION PROBABILITIES")
print("=" * 50)

probabilities = model.predict_proba(X_test)[:, 1]

print("Probability calculation completed.")


# ============================================================
# ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print(f"ROC-AUC: {roc_auc:.4f}")


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []

print("\n" + "=" * 50)
print("THRESHOLD ANALYSIS")
print("=" * 50)

for threshold in THRESHOLDS:

    # Convert probability into class prediction
    y_pred = (
        probabilities >= threshold
    ).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # False Positive Rate
    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    # False Negative Rate
    fnr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

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


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print(
    results_df[
        [
            "Threshold",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "FPR",
            "FNR"
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE CSV
# ============================================================

output_csv = os.path.join(
    OUTPUT_DIR,
    "threshold_analysis.csv"
)

results_df.to_csv(
    output_csv,
    index=False
)

print("\nSaved:", output_csv)


# ============================================================
# FIND BEST THRESHOLD BY F1
# ============================================================

best_f1_row = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n" + "=" * 50)
print("BEST THRESHOLD ACCORDING TO F1")
print("=" * 50)

print(
    f"Threshold : {best_f1_row['Threshold']:.2f}"
)

print(
    f"Accuracy  : {best_f1_row['Accuracy']:.4f}"
)

print(
    f"Precision : {best_f1_row['Precision']:.4f}"
)

print(
    f"Recall    : {best_f1_row['Recall']:.4f}"
)

print(
    f"F1        : {best_f1_row['F1']:.4f}"
)

print(
    f"FPR       : {best_f1_row['FPR']:.4f}"
)

print(
    f"FNR       : {best_f1_row['FNR']:.4f}"
)


# ============================================================
# FIND BEST THRESHOLD FOR HIGH RECALL
# ============================================================

high_recall = results_df[
    results_df["Recall"] >= 0.90
]

print("\n" + "=" * 50)
print("THRESHOLDS WITH RECALL >= 90%")
print("=" * 50)

if len(high_recall) > 0:

    print(
        high_recall[
            [
                "Threshold",
                "Precision",
                "Recall",
                "F1",
                "FPR",
                "FNR"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "No tested threshold achieved Recall >= 90%."
    )


# ============================================================
# CREATE PLOT
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["F1"],
    marker="o",
    label="F1 Score"
)

plt.plot(
    results_df["Threshold"],
    results_df["FPR"],
    marker="o",
    label="False Positive Rate"
)

plt.plot(
    results_df["Threshold"],
    results_df["FNR"],
    marker="o",
    label="False Negative Rate"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Metric Value")

plt.title(
    "Random Forest Threshold Analysis"
)

plt.xticks(THRESHOLDS)

plt.ylim(0, 1.05)

plt.grid(True)

plt.legend()

plt.tight_layout()


# ============================================================
# SAVE PLOT
# ============================================================

output_plot = os.path.join(
    OUTPUT_DIR,
    "threshold_analysis.png"
)

plt.savefig(
    output_plot,
    dpi=300
)

plt.close()

print("\nSaved:", output_plot)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 50)
print("THRESHOLD ANALYSIS COMPLETED")
print("=" * 50)