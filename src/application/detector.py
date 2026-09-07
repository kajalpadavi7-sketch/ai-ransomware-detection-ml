import os
import joblib
import pefile
import pandas as pd
from datetime import datetime, timezone
# ============================================================
# MODEL PATHS
# ============================================================

RF_MODEL_PATH = "results/models/standard/random_forest.joblib"
ISOLATION_MODEL_PATH = "results/unknown_detection/isolation_forest.joblib"


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
# LOAD MODELS
# ============================================================

print("Loading Random Forest model...")
rf_model = joblib.load(RF_MODEL_PATH)

print("Loading Isolation Forest model...")
isolation_model = joblib.load(ISOLATION_MODEL_PATH)

print("Models loaded successfully.")


# ============================================================
# EXTRACT PE FEATURES
# ============================================================
def extract_features(file_path):

    pe = pefile.PE(file_path)

    # ========================================================
    # SIZE
    # ========================================================

    size = os.path.getsize(file_path)

    # ========================================================
    # IMPORTS
    # ========================================================

    apis = []
    dlls = []

    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):

        for entry in pe.DIRECTORY_ENTRY_IMPORT:

            try:
                dll_name = entry.dll.decode(
                    "utf-8",
                    errors="ignore"
                )
            except Exception:
                dll_name = str(entry.dll)

            dlls.append(dll_name)

            for imp in entry.imports:

                if imp.name:

                    try:
                        api_name = imp.name.decode(
                            "utf-8",
                            errors="ignore"
                        )
                    except Exception:
                        api_name = str(imp.name)

                    apis.append(api_name)

    # ========================================================
    # COUNTS
    # ========================================================

    api_count = len(apis)

    dll_count = len(dlls)

    unique_api_count = len(set(apis))

    unique_dll_count = len(set(dlls))

    # ========================================================
    # ENTROPY
    # ========================================================

    entropy = 0.0

    try:

        entropies = []

        for section in pe.sections:

            entropies.append(
                section.get_entropy()
            )

        if entropies:

            entropy = sum(entropies) / len(entropies)

    except Exception:

        entropy = 0.0

    # ========================================================
    # PACKED
    # ========================================================

    packed = 0

    if entropy >= 7.0:
        packed = 1

    # ========================================================
    # PE YEAR
    # ========================================================

    year = 0

    try:

        timestamp = pe.FILE_HEADER.TimeDateStamp

        year = datetime.fromtimestamp(
            timestamp,
            tz=timezone.utc
        ).year

    except Exception:

        year = 0

    # ========================================================
    # FEATURE DICTIONARY
    # ========================================================

    features = {

        "API_Count": api_count,

        "DLL_Count": dll_count,

        "Unique_API_Count": unique_api_count,

        "Unique_DLL_Count": unique_dll_count,

        "Size": size,

        "Packed": packed,

        "Entropy": entropy,

        "Year": year

    }

    pe.close()

    return features

# ============================================================
# DETECTION
# ============================================================

def detect_file(file_path):

    # --------------------------------------------------------
    # Extract Features
    # --------------------------------------------------------

    features = extract_features(file_path)

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    X = pd.DataFrame(
        [[features[f] for f in FEATURES]],
        columns=FEATURES
    )

    # ========================================================
    # RANDOM FOREST CLASSIFICATION
    # ========================================================

    ransomware_probability = rf_model.predict_proba(X)[0][1]

    ransomware_probability_percent = (
        ransomware_probability * 100
    )

    # --------------------------------------------------------
    # RF Classification
    # --------------------------------------------------------

    if ransomware_probability >= 0.50:

        rf_classification = "RANSOMWARE"

    else:

        rf_classification = "BENIGN"

    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    anomaly_prediction = isolation_model.predict(X)[0]

    anomaly_score = isolation_model.decision_function(X)[0]

    if anomaly_prediction == -1:

        is_anomalous = True

        novelty_status = (
            "Potentially Novel / Anomalous"
        )

    else:

        is_anomalous = False

        novelty_status = (
            "Known Pattern / Normal"
        )

    # ========================================================
    # HYBRID DECISION
    # ========================================================
    #
    # RF = ransomware classification
    # Isolation Forest = novelty/anomaly detection
    #
    # Novel ransomware is declared ONLY when:
    #
    # RF says RANSOMWARE
    # AND
    # Isolation Forest says ANOMALOUS
    #
    # ========================================================

    if rf_classification == "RANSOMWARE":

        if is_anomalous:

            final_classification = (
                "POTENTIALLY NOVEL RANSOMWARE"
            )

            detection_type = (
                "Ransomware + Novelty Detection"
            )

        else:

            final_classification = (
                "RANSOMWARE"
            )

            detection_type = (
                "Ransomware Detection"
            )

    else:

        if is_anomalous:

            final_classification = (
                "BENIGN - ANOMALOUS"
            )

            detection_type = (
                "Anomalous File Detection"
            )

        else:

            final_classification = (
                "BENIGN"
            )

            detection_type = (
                "Benign Classification"
            )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    if final_classification == "POTENTIALLY NOVEL RANSOMWARE":

        risk_level = "CRITICAL"

    elif final_classification == "RANSOMWARE":

        if ransomware_probability >= 0.80:

            risk_level = "HIGH"

        else:

            risk_level = "MEDIUM"

    elif final_classification == "BENIGN - ANOMALOUS":

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    # ========================================================
    # CONFIDENCE
    # ========================================================

    if ransomware_probability >= 0.50:

        confidence = ransomware_probability_percent

    else:

        confidence = (
            100 - ransomware_probability_percent
        )

    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "classification": final_classification,

        "detection_type": detection_type,

        "ransomware_probability": round(
            ransomware_probability_percent,
            2
        ),

        "ransomware_confidence": round(
            confidence,
            2
        ),

        "novelty_status": novelty_status,

        "anomaly_score": round(
            float(anomaly_score),
            4
        ),

        "risk_level": risk_level,

        "features": features
    }

    return result