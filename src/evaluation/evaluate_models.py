import pandas as pd
import numpy as np
import os
import joblib

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)


# ==================================================
# PATHS
# ==================================================

TEST_FILE = "data/processed/test_unseen_family.csv"

MODEL_DIR = "results/models"

RESULT_DIR = "results/evaluation"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ==================================================
# FEATURES
# ==================================================

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


# ==================================================
# LOAD TEST DATA
# ==================================================

print("========================================")
print("LOADING TEST DATA")
print("========================================")

test_df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

X_test = test_df[FEATURES].copy()

y_test = test_df[TARGET].astype(int)

print("Test shape:", test_df.shape)

print("\nClass distribution:")
print(y_test.value_counts())


# ==================================================
# MODEL FILES
# ==================================================

model_files = {

    "Logistic Regression":
        "logistic_regression.joblib",

    "Decision Tree":
        "decision_tree.joblib",

    "Random Forest":
        "random_forest.joblib",

    "XGBoost":
        "xgboost.joblib",

    "LightGBM":
        "lightgbm.joblib"
}


# ==================================================
# CONFUSION MATRICES
# ==================================================

print("\n========================================")
print("CONFUSION MATRICES")
print("========================================")


for name, filename in model_files.items():

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )

    model = joblib.load(
        model_path
    )

    y_pred = model.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n----------------------------------------")
    print(name)
    print("----------------------------------------")

    print(cm)

    print("\nTN:", cm[0][0])
    print("FP:", cm[0][1])
    print("FN:", cm[1][0])
    print("TP:", cm[1][1])

    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Benign", "Actual Ransomware"],
        columns=["Predicted Benign", "Predicted Ransomware"]
    )

    output_file = os.path.join(
        RESULT_DIR,
        name.lower().replace(" ", "_") + "_confusion_matrix.csv"
    )

    cm_df.to_csv(
        output_file
    )

    print("Saved:", output_file)


# ==================================================
# CLASSIFICATION REPORTS
# ==================================================

print("\n========================================")
print("CLASSIFICATION REPORTS")
print("========================================")


for name, filename in model_files.items():

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )

    model = joblib.load(
        model_path
    )

    y_pred = model.predict(
        X_test
    )

    print("\n----------------------------------------")
    print(name)
    print("----------------------------------------")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Benign",
                "Ransomware"
            ],
            digits=4
        )
    )


# ==================================================
# ROC CURVE
# ==================================================

print("\n========================================")
print("CREATING ROC CURVE")
print("========================================")

plt.figure(
    figsize=(9, 7)
)

for name, filename in model_files.items():

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )

    model = joblib.load(
        model_path
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.4f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Unseen Ransomware Families"
)

plt.legend()

plt.grid(
    True
)

roc_file = os.path.join(
    RESULT_DIR,
    "roc_curve.png"
)

plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("ROC curve saved:", roc_file)


# ==================================================
# PRECISION-RECALL CURVE
# ==================================================

print("\n========================================")
print("CREATING PRECISION-RECALL CURVE")
print("========================================")

plt.figure(
    figsize=(9, 7)
)

for name, filename in model_files.items():

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )

    model = joblib.load(
        model_path
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    ap = average_precision_score(
        y_test,
        y_prob
    )

    plt.plot(
        recall,
        precision,
        label=f"{name} (AP = {ap:.4f})"
    )


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve - Unseen Ransomware Families"
)

plt.legend()

plt.grid(
    True
)

pr_file = os.path.join(
    RESULT_DIR,
    "precision_recall_curve.png"
)

plt.savefig(
    pr_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Precision-Recall curve saved:",
    pr_file
)


# ==================================================
# COMPLETED
# ==================================================

print("\n========================================")
print("EVALUATION COMPLETED")
print("========================================")

print("Results directory:")
print(RESULT_DIR)