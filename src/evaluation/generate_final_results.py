# ============================================================
# FINAL RESEARCH RESULTS GENERATOR
# ============================================================
#
# This script combines the results of:
#
# 1. Standard model evaluation
# 2. Standard Random Forest evaluation
# 3. Unseen-family evaluation
# 4. Threshold analysis
#
# No model training is performed here.
# ============================================================

import pandas as pd
import os


# ============================================================
# FILE PATHS
# ============================================================

STANDARD_RESULTS = "results/standard_model_comparison.csv"

UNSEEN_OVERALL = (
    "results/evaluation/unseen_overall_metrics.csv"
)

UNSEEN_FAMILY = (
    "results/evaluation/unseen_family_performance.csv"
)

THRESHOLD_RESULTS = (
    "results/evaluation/threshold_analysis.csv"
)

OUTPUT_DIR = "results/final"

FINAL_MODEL_RESULTS = (
    "results/final/final_model_comparison.csv"
)

FINAL_SUMMARY = (
    "results/final/final_research_summary.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 1. LOAD STANDARD MODEL RESULTS
# ============================================================

print("=" * 70)
print("LOADING STANDARD MODEL RESULTS")
print("=" * 70)

standard_df = pd.read_csv(
    STANDARD_RESULTS
)

print(
    "Standard model results shape:",
    standard_df.shape
)

print("\nModels:")
print(
    standard_df["Model"].tolist()
)


# ============================================================
# 2. DISPLAY STANDARD RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STANDARD MODEL PERFORMANCE")
print("=" * 70)

standard_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "FPR",
    "FNR",
    "Training_Time"
]

print(
    standard_df[
        standard_columns
    ].to_string(index=False)
)


# ============================================================
# 3. SAVE CLEAN STANDARD MODEL COMPARISON
# ============================================================

standard_clean = standard_df[
    standard_columns
].copy()

standard_clean = standard_clean.sort_values(
    by="F1",
    ascending=False
)

standard_clean.to_csv(
    FINAL_MODEL_RESULTS,
    index=False
)

print(
    "\nSaved standard comparison:"
)

print(
    FINAL_MODEL_RESULTS
)


# ============================================================
# 4. FIND BEST MODEL
# ============================================================

best_model_row = standard_clean.iloc[0]

best_model = best_model_row["Model"]

best_accuracy = best_model_row["Accuracy"]
best_precision = best_model_row["Precision"]
best_recall = best_model_row["Recall"]
best_f1 = best_model_row["F1"]
best_auc = best_model_row["ROC_AUC"]
best_fpr = best_model_row["FPR"]
best_fnr = best_model_row["FNR"]


print("\n" + "=" * 70)
print("BEST STANDARD MODEL")
print("=" * 70)

print(
    "Best model according to F1:",
    best_model
)

print(
    f"Accuracy : {best_accuracy:.4f}"
)

print(
    f"Precision: {best_precision:.4f}"
)

print(
    f"Recall   : {best_recall:.4f}"
)

print(
    f"F1       : {best_f1:.4f}"
)

print(
    f"ROC-AUC  : {best_auc:.4f}"
)


# ============================================================
# 5. LOAD UNSEEN OVERALL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LOADING UNSEEN-FAMILY OVERALL RESULTS")
print("=" * 70)

unseen_overall_df = pd.read_csv(
    UNSEEN_OVERALL
)

print(
    unseen_overall_df.to_string(
        index=False
    )
)


# ============================================================
# 6. EXTRACT UNSEEN OVERALL METRICS
# ============================================================

unseen_row = unseen_overall_df.iloc[0]

unseen_accuracy = unseen_row["Accuracy"]
unseen_precision = unseen_row["Precision"]
unseen_recall = unseen_row["Recall"]
unseen_f1 = unseen_row["F1"]
unseen_auc = unseen_row["ROC_AUC"]
unseen_fpr = unseen_row["FPR"]
unseen_fnr = unseen_row["FNR"]


# ============================================================
# 7. LOAD UNSEEN FAMILY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LOADING FAMILY-LEVEL RESULTS")
print("=" * 70)

family_df = pd.read_csv(
    UNSEEN_FAMILY
)

print(
    "Number of family records:",
    len(family_df)
)


# ============================================================
# 8. CALCULATE FAMILY-LEVEL STATISTICS
# ============================================================

number_of_unseen_families = (
    family_df["Family"].nunique()
)

mean_family_recall = (
    family_df["Recall"].mean()
)

median_family_recall = (
    family_df["Recall"].median()
)

minimum_family_recall = (
    family_df["Recall"].min()
)

maximum_family_recall = (
    family_df["Recall"].max()
)


print("\nFamily-level statistics:")

print(
    "Unseen families:",
    number_of_unseen_families
)

print(
    f"Mean recall   : {mean_family_recall:.4f}"
)

print(
    f"Median recall : {median_family_recall:.4f}"
)

print(
    f"Minimum recall: {minimum_family_recall:.4f}"
)

print(
    f"Maximum recall: {maximum_family_recall:.4f}"
)


# ============================================================
# 9. LOAD THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("LOADING THRESHOLD ANALYSIS")
print("=" * 70)

threshold_df = pd.read_csv(
    THRESHOLD_RESULTS
)

print(
    threshold_df.to_string(
        index=False
    )
)


# ============================================================
# 10. FIND BEST THRESHOLD BY F1
# ============================================================

best_threshold_row = (
    threshold_df.loc[
        threshold_df["F1"].idxmax()
    ]
)

best_threshold = (
    best_threshold_row["Threshold"]
)

threshold_accuracy = (
    best_threshold_row["Accuracy"]
)

threshold_precision = (
    best_threshold_row["Precision"]
)

threshold_recall = (
    best_threshold_row["Recall"]
)

threshold_f1 = (
    best_threshold_row["F1"]
)

threshold_fpr = (
    best_threshold_row["FPR"]
)

threshold_fnr = (
    best_threshold_row["FNR"]
)


print("\n" + "=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {threshold_accuracy:.4f}"
)

print(
    f"Precision : {threshold_precision:.4f}"
)

print(
    f"Recall    : {threshold_recall:.4f}"
)

print(
    f"F1        : {threshold_f1:.4f}"
)

print(
    f"FPR       : {threshold_fpr:.4f}"
)

print(
    f"FNR       : {threshold_fnr:.4f}"
)


# ============================================================
# 11. CREATE FINAL RESEARCH SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CREATING FINAL RESEARCH SUMMARY")
print("=" * 70)


summary = pd.DataFrame({

    "Experiment": [
        "Standard Test",
        "Unseen-Family Test",
        "Threshold Optimization"
    ],

    "Model": [
        best_model,
        "Random Forest",
        "Random Forest"
    ],

    "Accuracy": [
        best_accuracy,
        unseen_accuracy,
        threshold_accuracy
    ],

    "Precision": [
        best_precision,
        unseen_precision,
        threshold_precision
    ],

    "Recall": [
        best_recall,
        unseen_recall,
        threshold_recall
    ],

    "F1": [
        best_f1,
        unseen_f1,
        threshold_f1
    ],

    "ROC_AUC": [
        best_auc,
        unseen_auc,
        None
    ],

    "FPR": [
        best_fpr,
        unseen_fpr,
        threshold_fpr
    ],

    "FNR": [
        best_fnr,
        unseen_fnr,
        threshold_fnr
    ],

    "Threshold": [
        0.50,
        0.50,
        best_threshold
    ]
})


# ============================================================
# 12. SAVE FINAL SUMMARY
# ============================================================

summary.to_csv(
    FINAL_SUMMARY,
    index=False
)


# ============================================================
# 13. DISPLAY FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESEARCH SUMMARY")
print("=" * 70)

print(
    summary.to_string(index=False)
)


# ============================================================
# 14. FAMILY-LEVEL SUMMARY FILE
# ============================================================

family_summary = pd.DataFrame({

    "Metric": [
        "Number of unseen families",
        "Mean family recall",
        "Median family recall",
        "Minimum family recall",
        "Maximum family recall"
    ],

    "Value": [
        number_of_unseen_families,
        mean_family_recall,
        median_family_recall,
        minimum_family_recall,
        maximum_family_recall
    ]
})


FAMILY_SUMMARY_FILE = (
    "results/final/unseen_family_summary.csv"
)


family_summary.to_csv(
    FAMILY_SUMMARY_FILE,
    index=False
)


# ============================================================
# 15. COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULTS GENERATION COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(
    "1.",
    FINAL_MODEL_RESULTS
)

print(
    "2.",
    FINAL_SUMMARY
)

print(
    "3.",
    FAMILY_SUMMARY_FILE
)

print("\nBest model:")
print(best_model)

print(
    f"Standard F1: {best_f1:.4f}"
)

print(
    f"Unseen-family overall F1: {unseen_f1:.4f}"
)

print(
    f"Mean unseen-family recall: "
    f"{mean_family_recall:.4f}"
)

print(
    f"Best threshold: "
    f"{best_threshold:.2f}"
)

print("\nAll final results are ready for research analysis.")