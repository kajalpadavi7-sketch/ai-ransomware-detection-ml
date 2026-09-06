import pandas as pd
import os

INPUT_FILE = "data/processed/api_features.csv"

TRAIN_FILE = "data/processed/train_unseen_family.csv"
TEST_FILE = "data/processed/test_unseen_family.csv"

# Percentage of ransomware families kept completely unseen
UNSEEN_FAMILY_PERCENT = 20

# Reproducibility
RANDOM_STATE = 42


print("========================================")
print("LOADING FEATURE DATASET")
print("========================================")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# Separate benign and ransomware samples
# --------------------------------------------------

benign = df[df["Label"] == 0].copy()
ransomware = df[df["Label"] == 1].copy()

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print("Benign samples:", len(benign))
print("Ransomware samples:", len(ransomware))

print("Ransomware families:", ransomware["Family"].nunique())


# --------------------------------------------------
# Get ransomware families
# --------------------------------------------------

families = ransomware["Family"].dropna().unique()

print("\nTotal families:", len(families))


# --------------------------------------------------
# Randomly select families for unseen test set
# --------------------------------------------------

import numpy as np

np.random.seed(RANDOM_STATE)

num_unseen = int(len(families) * UNSEEN_FAMILY_PERCENT / 100)

unseen_families = np.random.choice(
    families,
    size=num_unseen,
    replace=False
)

unseen_families = set(unseen_families)


print("\n========================================")
print("UNSEEN FAMILY SELECTION")
print("========================================")

print("Percentage held out:", UNSEEN_FAMILY_PERCENT, "%")
print("Number of unseen families:", len(unseen_families))

print("\nFirst 20 unseen families:")

for family in sorted(list(unseen_families))[:20]:
    print(" -", family)


# --------------------------------------------------
# Create ransomware train/test split
# --------------------------------------------------

ransomware_test = ransomware[
    ransomware["Family"].isin(unseen_families)
].copy()

ransomware_train = ransomware[
    ~ransomware["Family"].isin(unseen_families)
].copy()


# --------------------------------------------------
# Split benign samples
# --------------------------------------------------

benign = benign.sample(
    frac=1,
    random_state=RANDOM_STATE
)

benign_test_size = int(len(benign) * 0.20)

benign_test = benign.iloc[:benign_test_size].copy()
benign_train = benign.iloc[benign_test_size:].copy()


# --------------------------------------------------
# Combine datasets
# --------------------------------------------------

train_df = pd.concat(
    [benign_train, ransomware_train],
    ignore_index=True
)

test_df = pd.concat(
    [benign_test, ransomware_test],
    ignore_index=True
)


# Shuffle final datasets

train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

test_df = test_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)


# --------------------------------------------------
# Save datasets
# --------------------------------------------------

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


# --------------------------------------------------
# Verification
# --------------------------------------------------

train_families = set(
    train_df[train_df["Label"] == 1]["Family"].dropna().unique()
)

test_families = set(
    test_df[test_df["Label"] == 1]["Family"].dropna().unique()
)

overlap = train_families.intersection(test_families)


print("\n========================================")
print("FINAL DATASET INFORMATION")
print("========================================")

print("\nTRAINING DATA")
print("--------------------")
print("Shape:", train_df.shape)

print("\nClass distribution:")
print(train_df["Label"].value_counts())

print("\nRansomware families:", len(train_families))


print("\nTEST DATA")
print("--------------------")
print("Shape:", test_df.shape)

print("\nClass distribution:")
print(test_df["Label"].value_counts())

print("\nRansomware families:", len(test_families))


# --------------------------------------------------
# Critical leakage check
# --------------------------------------------------

print("\n========================================")
print("UNSEEN FAMILY LEAKAGE CHECK")
print("========================================")

print("Families in both train and test:", len(overlap))

if len(overlap) == 0:
    print("✓ PASS: No ransomware family overlap")
    print("✓ Test ransomware families are completely unseen")
else:
    print("✗ ERROR: Family leakage detected!")
    print("Overlapping families:")
    print(overlap)


# --------------------------------------------------
# Save unseen family list
# --------------------------------------------------

unseen_family_file = "data/processed/unseen_families.txt"

with open(unseen_family_file, "w", encoding="utf-8") as file:
    for family in sorted(unseen_families):
        file.write(str(family) + "\n")


print("\n========================================")
print("SPLIT COMPLETED")
print("========================================")

print("Training dataset:")
print(TRAIN_FILE)

print("\nTest dataset:")
print(TEST_FILE)

print("\nUnseen family list:")
print(unseen_family_file)