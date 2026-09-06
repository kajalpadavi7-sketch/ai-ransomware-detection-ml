import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ==================================================
# PATHS
# ==================================================

TEST_FILE = "data/processed/test_unseen_family.csv"
MODEL_FILE = "results/models/random_forest.joblib"
OUTPUT_DIR = "results/shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ==================================================
# LOAD TEST DATA
# ==================================================

print("=" * 50)
print("LOADING TEST DATA")
print("=" * 50)

df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print("Test shape:", df.shape)

X_test = df[FEATURES]

print("Features:", FEATURES)

# ==================================================
# LOAD RANDOM FOREST
# ==================================================

print("\n" + "=" * 50)
print("LOADING RANDOM FOREST MODEL")
print("=" * 50)

model = joblib.load(MODEL_FILE)

print("Random Forest loaded successfully.")

# ==================================================
# CREATE SHAP EXPLAINER
# ==================================================

print("\n" + "=" * 50)
print("CREATING SHAP EXPLAINER")
print("=" * 50)

explainer = shap.TreeExplainer(
    model,
    feature_perturbation="tree_path_dependent"
)

print("SHAP TreeExplainer created.")

# ==================================================
# CALCULATE SHAP VALUES
# ==================================================

print("\n" + "=" * 50)
print("CALCULATING SHAP VALUES")
print("=" * 50)

# Use a sample for faster analysis
sample_size = min(1000, len(X_test))

X_sample = X_test.sample(
    n=sample_size,
    random_state=42
)

print("Samples used for SHAP:", len(X_sample))

shap_values = explainer.shap_values(X_sample)

# Handle binary classification output
if isinstance(shap_values, list):
    shap_values = shap_values[1]

elif len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

print("SHAP values calculated.")

# ==================================================
# GLOBAL FEATURE IMPORTANCE
# ==================================================

print("\n" + "=" * 50)
print("CALCULATING FEATURE IMPORTANCE")
print("=" * 50)

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Mean_Absolute_SHAP": abs(shap_values).mean(axis=0)
})

importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

print("\nFeature Importance:")
print(importance.to_string(index=False))

# ==================================================
# SAVE FEATURE IMPORTANCE
# ==================================================

importance_file = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv"
)

importance.to_csv(
    importance_file,
    index=False
)

print("\nSaved:", importance_file)

# ==================================================
# SHAP SUMMARY BAR PLOT
# ==================================================

print("\n" + "=" * 50)
print("CREATING SHAP FEATURE IMPORTANCE PLOT")
print("=" * 50)

plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False
)

plt.title(
    "Random Forest SHAP Feature Importance"
)

plt.tight_layout()

bar_plot = os.path.join(
    OUTPUT_DIR,
    "shap_feature_importance.png"
)

plt.savefig(
    bar_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", bar_plot)

# ==================================================
# SHAP SUMMARY PLOT
# ==================================================

print("\n" + "=" * 50)
print("CREATING SHAP SUMMARY PLOT")
print("=" * 50)

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.title(
    "Random Forest SHAP Summary Plot"
)

plt.tight_layout()

summary_plot = os.path.join(
    OUTPUT_DIR,
    "shap_summary_plot.png"
)

plt.savefig(
    summary_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", summary_plot)

# ==================================================
# COMPLETED
# ==================================================

print("\n" + "=" * 50)
print("SHAP ANALYSIS COMPLETED")
print("=" * 50)

print("Results saved in:")
print(OUTPUT_DIR)