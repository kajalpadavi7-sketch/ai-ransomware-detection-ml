# ============================================================
# STANDARD TRAIN / VALIDATION / TEST SPLIT
# ============================================================

import pandas as pd
import os

from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = "data/processed/ml_features.csv"

TRAIN_FILE = "data/processed/standard_train.csv"
VALIDATION_FILE = "data/processed/standard_validation.csv"
TEST_FILE = "data/processed/standard_test.csv"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING ML FEATURE DATASET")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Dataset shape:", df.shape)


# ============================================================
# CHECK DATA
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nClass distribution:")

print(
    df["Label"].value_counts()
)

print("\nClass percentages:")

print(
    (df["Label"].value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# FEATURES AND TARGET
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
# PREPARE DATA
# ============================================================

X = df[FEATURES]

y = df[TARGET].astype(int)


# ============================================================
# FIRST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRAINING AND TEMPORARY DATA")
print("=" * 60)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE + TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)


# ============================================================
# SECOND SPLIT
# ============================================================

print("\n" + "=" * 60)
print("CREATING VALIDATION AND TEST DATA")
print("=" * 60)

# Half of the remaining 30% = 15% validation
# Half of the remaining 30% = 15% test

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)


# ============================================================
# CREATE DATAFRAMES
# ============================================================

train_df = X_train.copy()
train_df[TARGET] = y_train.values

validation_df = X_validation.copy()
validation_df[TARGET] = y_validation.values

test_df = X_test.copy()
test_df[TARGET] = y_test.values


# ============================================================
# SAVE DATASETS
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

validation_df.to_csv(
    VALIDATION_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("STANDARD SPLIT RESULTS")
print("=" * 60)

print("\nTraining:")
print("Shape:", train_df.shape)
print(train_df[TARGET].value_counts())

print("\nValidation:")
print("Shape:", validation_df.shape)
print(validation_df[TARGET].value_counts())

print("\nTest:")
print("Shape:", test_df.shape)
print(test_df[TARGET].value_counts())


# ============================================================
# CHECK PERCENTAGES
# ============================================================

total = len(df)

print("\n" + "=" * 60)
print("SPLIT PERCENTAGES")
print("=" * 60)

print(
    "Training   :",
    round(len(train_df) / total * 100, 2),
    "%"
)

print(
    "Validation :",
    round(len(validation_df) / total * 100, 2),
    "%"
)

print(
    "Test       :",
    round(len(test_df) / total * 100, 2),
    "%"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("STANDARD SPLIT COMPLETED")
print("=" * 60)

print("\nSaved files:")

print(TRAIN_FILE)
print(VALIDATION_FILE)
print(TEST_FILE)