import pandas as pd
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


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = "data/processed/standard_train.csv"
VALIDATION_FILE = "data/processed/standard_validation.csv"
TEST_FILE = "data/processed/standard_test.csv"

RESULT_FILE = "results/standard_model_comparison.csv"

MODEL_DIR = "results/models/standard"


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

TARGET = "Label"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs("results", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING STANDARD DATASETS")
print("=" * 60)

train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

validation_df = pd.read_csv(
    VALIDATION_FILE,
    low_memory=False
)

test_df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print("Training shape   :", train_df.shape)
print("Validation shape :", validation_df.shape)
print("Test shape       :", test_df.shape)


# ============================================================
# PREPARE DATA
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET].astype(int)

X_val = validation_df[FEATURES]
y_val = validation_df[TARGET].astype(int)

X_test = test_df[FEATURES]
y_test = test_df[TARGET].astype(int)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print("\nTraining:")
print(y_train.value_counts())

print("\nValidation:")
print(y_val.value_counts())

print("\nTest:")
print(y_test.value_counts())


# ============================================================
# MODELS
# ============================================================

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


# ============================================================
# TRAINING
# ============================================================

results = []

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)


for name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", name)
    print("-" * 60)

    start_time = time.time()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    # ========================================================
    # VALIDATION
    # ========================================================

    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]

    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )

    val_precision = precision_score(
        y_val,
        val_pred,
        zero_division=0
    )

    val_recall = recall_score(
        y_val,
        val_pred,
        zero_division=0
    )

    val_f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    val_roc_auc = roc_auc_score(
        y_val,
        val_prob
    )

    print("\nValidation F1:", round(val_f1, 4))


    # ========================================================
    # FINAL TEST
    # ========================================================

    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        test_pred
    )

    precision = precision_score(
        y_test,
        test_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        test_prob
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_pred
    ).ravel()


    # ========================================================
    # FPR / FNR
    # ========================================================

    fpr = fp / (fp + tn)

    fnr = fn / (fn + tp)


    # ========================================================
    # DISPLAY TEST RESULTS
    # ========================================================

    print("\nFINAL TEST RESULTS")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("FPR      :", round(fpr, 4))
    print("FNR      :", round(fnr, 4))

    print("Training :", round(training_time, 2), "seconds")

    print("\nConfusion Matrix")
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_filename = (
        name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    model_path = os.path.join(
        MODEL_DIR,
        model_filename
    )

    joblib.dump(
        model,
        model_path
    )

    print("\nModel saved:", model_path)


    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append({

        "Model": name,

        "Validation_Accuracy": val_accuracy,
        "Validation_Precision": val_precision,
        "Validation_Recall": val_recall,
        "Validation_F1": val_f1,
        "Validation_ROC_AUC": val_roc_auc,

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


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("STANDARD MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(index=False)
)


print("\n" + "=" * 60)
print("STANDARD EVALUATION COMPLETED")
print("=" * 60)

print("Results saved:")
print(RESULT_FILE)

print("\nModels saved:")
print(MODEL_DIR)

print("\nBest model according to F1:")

print(
    results_df.iloc[0]["Model"]
)