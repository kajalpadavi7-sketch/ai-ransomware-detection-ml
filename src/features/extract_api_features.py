import pandas as pd
import json
import os


# ============================================================
# PATHS
# ============================================================

BENIGN_FILE = "data/processed/benign_clean.csv"
RANSOMWARE_FILE = "data/processed/ransomware_clean.csv"

OUTPUT_FILE = "data/processed/api_features.csv"


# ============================================================
# LOAD JSON
# ============================================================

def load_json_features(filepath):

    # Convert dataset/d0/file.json
    # into data/raw/RanDS/dataset/d0/file.json

    full_path = os.path.join(
        "data/raw/RanDS",
        filepath
    )

    if not os.path.exists(full_path):
        return {}

    try:

        with open(
            full_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:
        return {}


# ============================================================
# EXTRACT API INFORMATION
# ============================================================

def extract_api_features(json_data):

    imports = json_data.get("imports", {})

    apis = []
    dlls = []

    for dll, functions in imports.items():

        dlls.append(dll)

        if isinstance(functions, list):

            for function in functions:
                apis.append(function)

    return apis, dlls


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(df, label, is_ransomware=False):

    records = []

    for index, row in df.iterrows():

        json_data = load_json_features(
            row["Filepath"]
        )

        apis, dlls = extract_api_features(
            json_data
        )

        record = {

            "SHA256": row["SHA256"],

            "APIs": "|".join(apis),

            "DLLs": "|".join(dlls),

            "API_Count": len(apis),

            "DLL_Count": len(dlls),

            "Unique_API_Count": len(set(apis)),

            "Unique_DLL_Count": len(set(dlls)),

            "Size": row["Size in bytes"],

            "Packed": row["Packed"],

            "Entropy": row["Entropy"],

            "Year": row["Year"],

            "Label": label
        }

        # ----------------------------------------------------
        # Preserve ransomware family
        # ----------------------------------------------------

        if is_ransomware:

            record["Family"] = row["Family"]

        else:

            record["Family"] = "Benign"

        records.append(record)

        if (index + 1) % 10000 == 0:

            print(
                f"Processed {index + 1} samples"
            )

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

print("Loading cleaned Benign dataset...")

benign = pd.read_csv(
    BENIGN_FILE,
    low_memory=False
)

print(
    "Benign shape:",
    benign.shape
)


print("\nLoading cleaned Ransomware dataset...")

ransomware = pd.read_csv(
    RANSOMWARE_FILE,
    low_memory=False
)

print(
    "Ransomware shape:",
    ransomware.shape
)


# ============================================================
# PROCESS
# ============================================================

print("\nProcessing Benign dataset...")

benign_features = process_dataset(
    benign,
    label=0,
    is_ransomware=False
)


print("\nProcessing Ransomware dataset...")

ransomware_features = process_dataset(
    ransomware,
    label=1,
    is_ransomware=True
)


# ============================================================
# COMBINE
# ============================================================

print("\nCombining datasets...")

final_df = pd.concat(
    [
        benign_features,
        ransomware_features
    ],
    ignore_index=True
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n========================================")
print("FEATURE EXTRACTION COMPLETED")
print("========================================")

print(
    "Dataset shape:",
    final_df.shape
)

print(
    "Saved to:",
    OUTPUT_FILE
)

print("\nLabels:")

print(
    final_df["Label"].value_counts()
)

print("\nNumber of ransomware families:")

print(
    final_df[
        final_df["Label"] == 1
    ]["Family"].nunique()
)