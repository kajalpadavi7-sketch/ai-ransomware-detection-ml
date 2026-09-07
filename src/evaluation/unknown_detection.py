import os
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = "data/processed/train_unseen_family.csv"
TEST_FILE = "data/processed/test_unseen_family.csv"

OUTPUT_DIR = "results/unknown_detection"

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING TRAINING AND UNSEEN-FAMILY TEST DATA")
print("=" * 70)

train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

test_df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print("Training shape:", train_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# PREPARE RANSOMWARE DATA
# ============================================================

print("\n" + "=" * 70)
print("PREPARING RANSOMWARE DATA")
print("=" * 70)

# Only ransomware samples are used for the anomaly model
train_ransomware = train_df[
    train_df["Label"] == 1
].copy()

test_ransomware = test_df[
    test_df["Label"] == 1
].copy()

print(
    "Training ransomware samples:",
    len(train_ransomware)
)

print(
    "Unseen-family ransomware samples:",
    len(test_ransomware)
)

print(
    "Training ransomware families:",
    train_ransomware["Family"].nunique()
)

print(
    "Unseen ransomware families:",
    test_ransomware["Family"].nunique()
)


# ============================================================
# FEATURES
# ============================================================

X_train = train_ransomware[
    FEATURES
].copy()

X_test = test_ransomware[
    FEATURES
].copy()


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

X_train = X_train.fillna(
    X_train.median()
)

X_test = X_test.fillna(
    X_train.median()
)


# ============================================================
# ISOLATION FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ISOLATION FOREST")
print("=" * 70)

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "isolation_forest",
        IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
            n_jobs=-1
        )
    )
])


model.fit(X_train)

print(
    "Isolation Forest trained successfully."
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("ANALYZING UNSEEN RANSOMWARE")
print("=" * 70)

# Isolation Forest:
#
#  1  = normal
# -1  = anomaly

predictions = model.predict(X_test)

anomaly_score = model.decision_function(
    X_test
)


# ============================================================
# CONVERT TO UNKNOWN / NORMAL
# ============================================================

unknown_prediction = np.where(
    predictions == -1,
    1,
    0
)


# ============================================================
# ADD RESULTS
# ============================================================

results = test_ransomware[
    [
        "SHA256",
        "Family"
    ]
].copy()

results["Anomaly_Score"] = anomaly_score

results["Unknown"] = unknown_prediction


# ============================================================
# OVERALL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("UNKNOWN DETECTION RESULTS")
print("=" * 70)

total_samples = len(results)

unknown_samples = (
    results["Unknown"] == 1
).sum()

normal_samples = (
    results["Unknown"] == 0
).sum()

unknown_percentage = (
    unknown_samples / total_samples
) * 100

print(
    "Total unseen ransomware samples:",
    total_samples
)

print(
    "Detected as anomalous/unknown:",
    unknown_samples
)

print(
    "Detected as normal:",
    normal_samples
)

print(
    "Unknown detection percentage:",
    round(
        unknown_percentage,
        4
    ),
    "%"
)


# ============================================================
# FAMILY-LEVEL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("FAMILY-LEVEL UNKNOWN DETECTION")
print("=" * 70)

family_summary = (
    results
    .groupby("Family")
    .agg(
        Samples=("Unknown", "count"),
        Unknown_Samples=("Unknown", "sum")
    )
    .reset_index()
)

family_summary[
    "Unknown_Rate"
] = (
    family_summary["Unknown_Samples"]
    /
    family_summary["Samples"]
)

family_summary = family_summary.sort_values(
    "Unknown_Rate",
    ascending=False
)

print(
    family_summary.head(20).to_string(
        index=False
    )
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

mean_unknown_rate = (
    family_summary["Unknown_Rate"]
    .mean()
)

median_unknown_rate = (
    family_summary["Unknown_Rate"]
    .median()
)

print("\n" + "=" * 70)
print("FAMILY-LEVEL SUMMARY")
print("=" * 70)

print(
    "Number of unseen families:",
    len(family_summary)
)

print(
    "Mean unknown detection rate:",
    round(
        mean_unknown_rate,
        4
    )
)

print(
    "Median unknown detection rate:",
    round(
        median_unknown_rate,
        4
    )
)


# ============================================================
# SAVE SAMPLE-LEVEL RESULTS
# ============================================================

sample_output = os.path.join(
    OUTPUT_DIR,
    "unknown_detection_results.csv"
)

results.to_csv(
    sample_output,
    index=False
)

print(
    "\nSaved:",
    sample_output
)


# ============================================================
# SAVE FAMILY RESULTS
# ============================================================

family_output = os.path.join(
    OUTPUT_DIR,
    "family_unknown_detection.csv"
)

family_summary.to_csv(
    family_output,
    index=False
)

print(
    "Saved:",
    family_output
)


# ============================================================
# SAVE MODEL
# ============================================================

import joblib

model_output = os.path.join(
    OUTPUT_DIR,
    "isolation_forest.joblib"
)

joblib.dump(
    model,
    model_output
)

print(
    "Saved:",
    model_output
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2 UNKNOWN DETECTION COMPLETED")
print("=" * 70)

print(
    "Results directory:",
    OUTPUT_DIR
)