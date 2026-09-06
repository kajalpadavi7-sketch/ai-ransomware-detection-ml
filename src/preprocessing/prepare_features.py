import pandas as pd
import os


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = "data/processed/api_features.csv"
OUTPUT_FILE = "data/processed/ml_features.csv"


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading extracted feature dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Dataset shape:", df.shape)


# ============================================================
# 2. DISPLAY COLUMNS
# ============================================================

print("\nAvailable columns:")

for column in df.columns:
    print(" -", column)


# ============================================================
# 3. CHECK LABEL DISTRIBUTION
# ============================================================

print("\n========================================")
print("LABEL DISTRIBUTION")
print("========================================")

print(df["Label"].value_counts())

print("\nPercentage:")

print(
    (df["Label"].value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 4. CHECK RANSOMWARE FAMILIES
# ============================================================

print("\n========================================")
print("RANSOMWARE FAMILY INFORMATION")
print("========================================")

ransomware = df[df["Label"] == 1]

print(
    "Number of ransomware samples:",
    len(ransomware)
)

print(
    "Number of ransomware families:",
    ransomware["Family"].nunique()
)


# ============================================================
# 5. SELECT ML FEATURES
# ============================================================

features = [
    "API_Count",
    "DLL_Count",
    "Unique_API_Count",
    "Unique_DLL_Count",
    "Size",
    "Packed",
    "Entropy",
    "Year"
]

target = "Label"


print("\n========================================")
print("SELECTED ML FEATURES")
print("========================================")

for feature in features:
    print("✓", feature)


# ============================================================
# 6. CREATE ML DATASET
# ============================================================

ml_df = df[features + [target]].copy()


# ============================================================
# 7. CONVERT TO NUMERIC
# ============================================================

print("\nConverting features to numeric values...")

for column in features + [target]:

    ml_df[column] = pd.to_numeric(
        ml_df[column],
        errors="coerce"
    )


# ============================================================
# 8. CHECK MISSING VALUES
# ============================================================

print("\n========================================")
print("MISSING VALUES")
print("========================================")

print(
    ml_df.isnull().sum()
)


# ============================================================
# 9. REMOVE INVALID ROWS
# ============================================================

before = len(ml_df)

ml_df = ml_df.dropna()

after = len(ml_df)

print(
    "\nRows before cleaning:",
    before
)

print(
    "Rows after cleaning:",
    after
)

print(
    "Rows removed:",
    before - after
)


# ============================================================
# 10. SEPARATE X AND y
# ============================================================

X = ml_df[features]

y = ml_df[target].astype(int)


print("\n========================================")
print("ML DATA")
print("========================================")

print("X shape:", X.shape)

print("y shape:", y.shape)


# ============================================================
# 11. FINAL CLASS DISTRIBUTION
# ============================================================

print("\n========================================")
print("FINAL CLASS DISTRIBUTION")
print("========================================")

print(
    y.value_counts()
)

print("\nFinal percentages:")

print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 12. SAVE ML DATASET
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

ml_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n========================================")
print("PREPROCESSING COMPLETED")
print("========================================")

print(
    "Final dataset shape:",
    ml_df.shape
)

print(
    "Saved to:",
    OUTPUT_FILE
)