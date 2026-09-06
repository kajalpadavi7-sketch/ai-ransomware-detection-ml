import os
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix


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

df = pd.read_csv(TEST_FILE)

print(f"Test shape: {df.shape}")


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 50)
print("LOADING RANDOM FOREST MODEL")
print("=" * 50)

model = joblib.load(MODEL_FILE)

print("Random Forest model loaded successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

X = df[FEATURES]
y = df["Label"]

df["Prediction"] = model.predict(X)


# ============================================================
# FILTER RANSOMWARE SAMPLES
# ============================================================

ransomware_df = df[df["Label"] == 1].copy()

print(f"\nTotal ransomware samples in test set: {len(ransomware_df)}")


# ============================================================
# FAMILY-LEVEL PERFORMANCE
# ============================================================

results = []

for family, group in ransomware_df.groupby("Family"):

    y_true = group["Label"]
    y_pred = group["Prediction"]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    total = len(group)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    results.append({
        "Family": family,
        "Samples": total,
        "TP": tp,
        "FN": fn,
        "Recall": recall
    })


results_df = pd.DataFrame(results)


# ============================================================
# SORT BY RECALL
# ============================================================

results_df = results_df.sort_values(
    by="Recall",
    ascending=True
).reset_index(drop=True)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "unseen_family_performance.csv"
)

results_df.to_csv(output_file, index=False)


# ============================================================
# DISPLAY WORST PERFORMING FAMILIES
# ============================================================

print("\n" + "=" * 50)
print("WORST PERFORMING UNSEEN FAMILIES")
print("=" * 50)

print(
    results_df.head(20).to_string(index=False)
)


# ============================================================
# DISPLAY BEST PERFORMING FAMILIES
# ============================================================

print("\n" + "=" * 50)
print("BEST PERFORMING UNSEEN FAMILIES")
print("=" * 50)

print(
    results_df.tail(20)
    .sort_values("Recall", ascending=False)
    .to_string(index=False)
)


# ============================================================
# FAMILY-LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("FAMILY-LEVEL SUMMARY")
print("=" * 50)

print(f"Number of unseen families: {len(results_df)}")

print(
    f"Mean family recall: "
    f"{results_df['Recall'].mean():.4f}"
)

print(
    f"Median family recall: "
    f"{results_df['Recall'].median():.4f}"
)


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 50)
print("ANALYSIS COMPLETED")
print("=" * 50)

print(f"Saved: {output_file}")