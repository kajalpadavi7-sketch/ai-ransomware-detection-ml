import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# FILE PATHS
# ============================================================

TEST_FILE = "data/processed/standard_test.csv"

MODEL_FILE = "results/models/standard/random_forest.joblib"

THRESHOLD_FILE = "results/evaluation/standard/selected_threshold.txt"

OUTPUT_DIR = "results/evaluation/standard"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "standard_threshold_test_results.csv"
)

OUTPUT_REPORT = os.path.join(
    OUTPUT_DIR,
    "standard_threshold_test_report.txt"
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
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD SELECTED THRESHOLD
# ============================================================

print("=" * 60)
print("LOADING SELECTED VALIDATION THRESHOLD")
print("=" * 60)

with open(
    THRESHOLD_FILE,
    "r"
) as f:

    threshold_line = f.readline().strip()


best_threshold = float(
    threshold_line.split(":")[1].strip()
)

print(
    f"Selected threshold: {best_threshold:.2f}"
)

print(
    "Threshold was selected using validation data only."
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING STANDARD TEST DATA")
print("=" * 60)

df = pd.read_csv(
    TEST_FILE
)

print(
    "Test shape:",
    df.shape
)

print(
    "Test class distribution:"
)

print(
    df[TARGET].value_counts()
)


# ============================================================
# PREPARE DATA
# ============================================================

X_test = df[FEATURES]

y_test = df[TARGET]


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
# PREDICT PROBABILITIES
# ============================================================

print("\n" + "=" * 60)
print("PREDICTING TEST PROBABILITIES")
print("=" * 60)

probabilities = rf_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# APPLY SELECTED THRESHOLD
# ============================================================

y_pred = (
    probabilities >= best_threshold
).astype(int)


# ============================================================
# METRICS
# ============================================================

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

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()


# ============================================================
# FPR / FNR
# ============================================================

if (tn + fp) > 0:

    fpr = fp / (tn + fp)

else:

    fpr = 0.0


if (tp + fn) > 0:

    fnr = fn / (tp + fn)

else:

    fnr = 0.0


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL STANDARD TEST RESULTS")
print("=" * 60)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print(
    f"FPR       : {fpr:.4f}"
)

print(
    f"FNR       : {fnr:.4f}"
)

print("\nConfusion Matrix:")

print(
    f"TN = {tn}"
)

print(
    f"FP = {fp}"
)

print(
    f"FN = {fn}"
)

print(
    f"TP = {tp}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Benign",
        "Ransomware"
    ],
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ============================================================
# SAVE RESULTS CSV
# ============================================================

results = pd.DataFrame([
    {

        "Threshold": best_threshold,

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

    }
])


results.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    OUTPUT_REPORT,
    "w"
) as f:

    f.write(
        "FINAL STANDARD TEST EVALUATION\n"
    )

    f.write(
        "================================\n\n"
    )

    f.write(
        f"Selected Threshold: {best_threshold:.2f}\n"
    )

    f.write(
        "Threshold Selection: Highest Validation F1\n\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall: {recall:.4f}\n"
    )

    f.write(
        f"F1: {f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC: {roc_auc:.4f}\n"
    )

    f.write(
        f"FPR: {fpr:.4f}\n"
    )

    f.write(
        f"FNR: {fnr:.4f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        f"TN = {tn}\n"
    )

    f.write(
        f"FP = {fp}\n"
    )

    f.write(
        f"FN = {fn}\n"
    )

    f.write(
        f"TP = {tp}\n\n"
    )

    f.write(
        "Classification Report:\n"
    )

    f.write(
        report
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 60)

print(
    f"Results saved to:\n{OUTPUT_CSV}"
)

print(
    f"Report saved to:\n{OUTPUT_REPORT}"
)

print(
    "\nIMPORTANT:"
)

print(
    "The test set was used only for final evaluation."
)

print(
    "The threshold was selected previously using validation data."
)