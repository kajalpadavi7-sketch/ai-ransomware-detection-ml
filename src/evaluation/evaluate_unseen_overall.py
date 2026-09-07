# ============================================================
# OVERALL PERFORMANCE ON UNSEEN RANSOMWARE FAMILIES
# ============================================================

import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

TEST_FILE = "data/processed/test_unseen_family.csv"

MODEL_FILE = "results/models/random_forest.joblib"

OUTPUT_FILE = "results/evaluation/unseen_overall_metrics.csv"


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 60)
print("LOADING UNSEEN-FAMILY TEST DATA")
print("=" * 60)

df = pd.read_csv(TEST_FILE)

print(f"Test dataset shape: {df.shape}")


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
y_test = df["Label"]


print("\nFeatures used:")
print(FEATURES)

print("\nClass distribution:")
print(y_test.value_counts())


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("LOADING RANDOM FOREST MODEL")
print("=" * 60)

model = joblib.load(MODEL_FILE)

print("Random Forest model loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

y_pred = model.predict(X_test)

# Probability of ransomware class (Label = 1)
y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

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
    y_prob
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()


# ============================================================
# FPR AND FNR
# ============================================================

fpr = fp / (fp + tn)

fnr = fn / (fn + tp)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("OVERALL UNSEEN-FAMILY PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")
print(f"ROC-AUC   : {roc_auc:.4f} ({roc_auc * 100:.2f}%)")

print("\nConfusion Matrix:")
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")

print(f"\nFPR       : {fpr:.4f} ({fpr * 100:.2f}%)")
print(f"FNR       : {fnr:.4f} ({fnr * 100:.2f}%)")


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["Random Forest"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1],
    "ROC_AUC": [roc_auc],
    "FPR": [fpr],
    "FNR": [fnr],
    "TN": [tn],
    "FP": [fp],
    "FN": [fn],
    "TP": [tp]
})

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)

print(f"Saved: {OUTPUT_FILE}")