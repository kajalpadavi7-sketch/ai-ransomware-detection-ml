import pandas as pd
import numpy as np
import os
import time
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ==================================================
# FILE PATHS
# ==================================================

TRAIN_FILE = "data/processed/train_unseen_family.csv"
TEST_FILE = "data/processed/test_unseen_family.csv"

RESULT_FILE = "results/model_comparison.csv"
MODEL_DIR = "results/models"


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
# CREATE DIRECTORIES
# ==================================================

os.makedirs("results", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==================================================
# LOAD DATA
# ==================================================

print("========================================")
print("LOADING TRAINING AND TEST DATA")
print("========================================")

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


# ==================================================
# PREPARE X AND Y
# ==================================================

X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].astype(int)

X_test = test_df[FEATURES].copy()
y_test = test_df[TARGET].astype(int)


print("\n========================================")
print("FEATURE INFORMATION")
print("========================================")

print("Features:")

for feature in FEATURES:
    print(" -", feature)

print("\nX_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ==================================================
# CLASS DISTRIBUTION
# ==================================================

print("\n========================================")
print("CLASS DISTRIBUTION")
print("========================================")

print("\nTraining:")
print(y_train.value_counts())

print("\nTesting:")
print(y_test.value_counts())


# ==================================================
# DEFINE MODELS
# ==================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
}


# ==================================================
# TRAIN MODELS
# ==================================================

results = []

print("\n========================================")
print("MODEL TRAINING")
print("========================================")


for name, model in models.items():

    print("\n----------------------------------------")
    print("Training:", name)
    print("----------------------------------------")

    start_time = time.time()

    # Train
    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    # False Positive Rate
    fpr = fp / (fp + tn)

    # False Negative Rate
    fnr = fn / (fn + tp)

    # --------------------------------------------------
    # Display metrics
    # --------------------------------------------------

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))
    print("FPR      :", round(fpr, 4))
    print("FNR      :", round(fnr, 4))
    print("Time     :", round(training_time, 2), "seconds")

    print("\nConfusion Matrix:")
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model_filename = name.lower().replace(" ", "_") + ".joblib"

    model_path = os.path.join(
        MODEL_DIR,
        model_filename
    )

    joblib.dump(
        model,
        model_path
    )

    print("Model saved:", model_path)

    # --------------------------------------------------
    # Store results
    # --------------------------------------------------

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": roc_auc,

        "FPR": fpr,

        "FNR": fnr,

        "Training_Time": training_time,

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "TP": tp
    })


# ==================================================
# SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ==================================================
# DISPLAY FINAL RESULTS
# ==================================================

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    results_df.to_string(index=False)
)


print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")

print("Results saved to:")
print(RESULT_FILE)

print("\nSaved models:")
print(MODEL_DIR)

print("\nBest model according to F1:")

print(
    results_df.iloc[0]["Model"]
)