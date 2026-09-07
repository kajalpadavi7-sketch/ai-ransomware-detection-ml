# ============================================================
# STANDARD TEST SET - CONFUSION MATRIX & CLASSIFICATION REPORT
# ============================================================

import pandas as pd
import joblib
import os

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ============================================================
# PATHS
# ============================================================

TEST_FILE = "data/processed/standard_test.csv"

MODEL_FILE = "results/models/standard/random_forest.joblib"

OUTPUT_DIR = "results/evaluation/standard"

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "standard_classification_report.txt"
)

CM_FILE = os.path.join(
    OUTPUT_DIR,
    "standard_confusion_matrix.csv"
)

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
# PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

y_pred = model.predict(X_test)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=["Actual Benign", "Actual Ransomware"],
    columns=["Predicted Benign", "Predicted Ransomware"]
)

cm_df.to_csv(
    CM_FILE
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Benign",
        "Ransomware"
    ],
    digits=4
)

print(report)

# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "STANDARD TEST SET - RANDOM FOREST\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"TN = {tn}\n"
        f"FP = {fp}\n"
        f"FN = {fn}\n"
        f"TP = {tp}\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        "=" * 60 + "\n"
    )

    file.write(report)

# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("STANDARD CONFUSION MATRIX ANALYSIS COMPLETED")
print("=" * 60)

print("\nSaved files:")

print(CM_FILE)
print(REPORT_FILE)