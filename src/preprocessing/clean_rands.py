import pandas as pd
import os

BASE_PATH = "data/raw/RanDS"

BENIGN_FILE = os.path.join(BASE_PATH, "Benign.csv")
RANSOMWARE_FILE = os.path.join(BASE_PATH, "Ransomware.csv")

OUTPUT_DIR = "data/processed"
BENIGN_OUTPUT = os.path.join(OUTPUT_DIR, "benign_clean.csv")
RANSOMWARE_OUTPUT = os.path.join(OUTPUT_DIR, "ransomware_clean.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD BENIGN DATASET
# ============================================================

print("Loading Benign dataset...")

benign = pd.read_csv(BENIGN_FILE)

print("Original benign shape:", benign.shape)

benign["Label"] = 0

print("\nBenign columns:")
print(benign.columns.tolist())


# ============================================================
# 2. LOAD RANSOMWARE DATASET
# ============================================================

print("\nLoading Ransomware dataset...")

ransomware = pd.read_csv(RANSOMWARE_FILE)

print("Original ransomware shape:", ransomware.shape)

print("\nOriginal ransomware columns:")
print(ransomware.columns.tolist())


# ============================================================
# 3. FIX RANSOMWARE COLUMN ORDER
# ============================================================
#
# Actual ransomware rows are:
#
# Arch, Packed, Entropy, Family, Year
#
# But CSV header says:
#
# Arch, Family, Packed, Entropy, Year
#
# Therefore we explicitly rebuild these columns.
# ============================================================

ransomware_clean = pd.DataFrame()

ransomware_clean["SHA256"] = ransomware["SHA256"]
ransomware_clean["SHA1"] = ransomware["SHA1"]
ransomware_clean["MD5"] = ransomware["MD5"]

ransomware_clean["Size in bytes"] = ransomware["Size in bytes"]
ransomware_clean["File extension"] = ransomware["File extension"]
ransomware_clean["Arch"] = ransomware["Arch"]

# IMPORTANT:
# Current Family column actually contains Packed
ransomware_clean["Packed"] = ransomware["Family"]

# Current Packed column actually contains Entropy
ransomware_clean["Entropy"] = ransomware["Packed"]

# Current Entropy column actually contains Family
ransomware_clean["Family"] = ransomware["Entropy"]

ransomware_clean["Year"] = ransomware["Year"]
ransomware_clean["Filepath"] = ransomware["Filepath"]

ransomware_clean["Label"] = 1


# ============================================================
# 4. CONVERT DATA TYPES
# ============================================================

benign["Size in bytes"] = pd.to_numeric(
    benign["Size in bytes"],
    errors="coerce"
)

benign["Packed"] = pd.to_numeric(
    benign["Packed"],
    errors="coerce"
)

benign["Entropy"] = pd.to_numeric(
    benign["Entropy"],
    errors="coerce"
)

benign["Year"] = pd.to_numeric(
    benign["Year"],
    errors="coerce"
)


ransomware_clean["Size in bytes"] = pd.to_numeric(
    ransomware_clean["Size in bytes"],
    errors="coerce"
)

ransomware_clean["Packed"] = pd.to_numeric(
    ransomware_clean["Packed"],
    errors="coerce"
)

ransomware_clean["Entropy"] = pd.to_numeric(
    ransomware_clean["Entropy"],
    errors="coerce"
)

ransomware_clean["Year"] = pd.to_numeric(
    ransomware_clean["Year"],
    errors="coerce"
)


# ============================================================
# 5. CHECK FAMILY VALUES
# ============================================================

print("\n========================================")
print("RANSOMWARE FAMILY CHECK")
print("========================================")

print("Number of unique families:",
      ransomware_clean["Family"].nunique())

print("\nFirst 10 families:")
print(ransomware_clean["Family"].head(10).tolist())

print("\nTop 20 families:")
print(ransomware_clean["Family"].value_counts().head(20))


# ============================================================
# 6. CHECK PACKED
# ============================================================

print("\n========================================")
print("PACKED CHECK")
print("========================================")

print(ransomware_clean["Packed"].value_counts())


# ============================================================
# 7. CHECK ENTROPY
# ============================================================

print("\n========================================")
print("ENTROPY CHECK")
print("========================================")

print(ransomware_clean["Entropy"].describe())


# ============================================================
# 8. SAVE CLEAN DATASETS
# ============================================================

benign.to_csv(BENIGN_OUTPUT, index=False)

ransomware_clean.to_csv(
    RANSOMWARE_OUTPUT,
    index=False
)

print("\n========================================")
print("CLEANING COMPLETED")
print("========================================")

print("Benign saved to:")
print(BENIGN_OUTPUT)

print("\nRansomware saved to:")
print(RANSOMWARE_OUTPUT)

print("\nFinal benign shape:",
      benign.shape)

print("Final ransomware shape:",
      ransomware_clean.shape)