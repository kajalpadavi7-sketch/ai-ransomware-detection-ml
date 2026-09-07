import os
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

RF_MODEL = "results/models/random_forest.joblib"

IF_MODEL = "results/unknown_detection/isolation_forest.joblib"

OUTPUT_DIR = "results/hybrid"

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
# LOAD MODELS
# ============================================================

print("=" * 70)
print("LOADING HYBRID DETECTION MODELS")
print("=" * 70)

rf_model = joblib.load(RF_MODEL)

isolation_model = joblib.load(IF_MODEL)

print("Random Forest loaded successfully.")
print("Isolation Forest loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def detect_ransomware(features):

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    X = pd.DataFrame(
        [features],
        columns=FEATURES
    )

    # --------------------------------------------------------
    # Random Forest prediction
    # --------------------------------------------------------

    rf_prediction = rf_model.predict(X)[0]

    rf_probability = rf_model.predict_proba(X)[0][1]

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    anomaly_prediction = isolation_model.predict(X)[0]

    anomaly_score = isolation_model.decision_function(X)[0]

    # --------------------------------------------------------
    # Convert anomaly result
    # --------------------------------------------------------

    if anomaly_prediction == -1:
        anomaly_status = "Potentially Novel / Anomalous"
    else:
        anomaly_status = "Normal Pattern"

    # --------------------------------------------------------
    # Final classification
    # --------------------------------------------------------

    if rf_prediction == 0:

        final_classification = "BENIGN"

        risk_level = "LOW"

    else:

        final_classification = "RANSOMWARE"

        if anomaly_prediction == -1:

            risk_level = "HIGH - POTENTIALLY NOVEL"

        else:

            risk_level = "HIGH"


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "Classification":
            final_classification,

        "Ransomware_Probability":
            round(
                float(rf_probability),
                4
            ),

        "Ransomware_Confidence":
            round(
                float(rf_probability) * 100,
                2
            ),

        "Novelty_Status":
            anomaly_status,

        "Anomaly_Score":
            round(
                float(anomaly_score),
                4
            ),

        "Risk_Level":
            risk_level
    }

    return result


# ============================================================
# TEST EXAMPLE
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("TESTING HYBRID DETECTION")
    print("=" * 70)

    # Example PE static features
    example_features = {

        "API_Count": 150,

        "DLL_Count": 20,

        "Unique_API_Count": 120,

        "Unique_DLL_Count": 18,

        "Size": 500000,

        "Packed": 1,

        "Entropy": 7.2,

        "Year": 2023
    }

    result = detect_ransomware(
        example_features
    )

    print("\nFINAL RESULT")
    print("-" * 50)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    print("\n" + "=" * 70)
    print("HYBRID DETECTION TEST COMPLETED")
    print("=" * 70)