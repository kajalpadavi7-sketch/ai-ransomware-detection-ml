# AI-Based Ransomware Detection Using Machine Learning

## 📌 Project Overview

This research project focuses on developing an Artificial Intelligence and
Machine Learning based framework for detecting ransomware.

The primary objective is to detect both:

1. Known ransomware samples that belong to ransomware families represented
   in the training data.
2. Previously unseen ransomware families using an unseen-family / novelty
   detection approach.

The project will investigate machine learning techniques for ransomware
detection using static features extracted from Windows Portable Executable
(PE) files.

---

## 🎯 Research Objectives

The main objectives of this research are:

- Detect ransomware using machine learning.
- Distinguish ransomware samples from benign software.
- Analyze static characteristics of PE files.
- Compare multiple machine learning algorithms.
- Evaluate the model using appropriate cybersecurity metrics.
- Test the model on ransomware families that were not included during
  training.
- Investigate novelty/anomaly detection for previously unseen samples.
- Analyze the important features contributing to model predictions.
- Evaluate the generalization capability of the proposed framework.

---

## 🔬 Research Problem

Traditional ransomware detection approaches may rely heavily on known
signatures and previously identified ransomware samples.

A major challenge is the detection of previously unseen ransomware or
ransomware families that were not represented in the training data.

This research therefore investigates a machine-learning framework that
combines conventional ransomware classification with an evaluation strategy
for previously unseen ransomware families.

The term "zero-day" in this project refers to an experimental
unseen-family/novelty-detection setting unless a genuine zero-day sample is
independently verified.

---

# 🏗️ Proposed Framework

The proposed research framework will follow the pipeline:

PE Sample
    ↓
Static Feature Representation
    ↓
Data Preprocessing
    ↓
Feature Analysis / Selection
    ↓
Machine Learning Model
    ↓
Known Ransomware Detection
    ↓
Unseen-Family / Novelty Detection
    ↓
Model Evaluation
    ↓
Explainable AI Analysis

---

# 📊 Dataset

## Primary Dataset: RanDS

The primary dataset used in this research is the RanDS dataset.

RanDS provides ransomware and benign PE-related datasets and includes
multiple ransomware families.

### Current Dataset

The first dataset being used in the implementation is:

**RanDS Static API Dataset**

Static API features are used to represent characteristics of PE files
without executing the files.

### Dataset Source

Official RanDS website:

https://ran-ds.com/home

The dataset will be used for research and experimentation according to the
dataset provider's terms and conditions.

### Dataset Storage

The downloaded dataset is stored locally in:

data/raw/

The original dataset is intentionally not uploaded to this GitHub repository.

---

# 🧩 Why Static API Features?

Portable Executable (PE) is the executable file format commonly used by
Windows applications.

Ransomware targeting Windows systems can be represented as PE files.

Static analysis allows characteristics of a PE file to be examined without
executing the file.

API-related features can provide useful information about the operations
and functionality referenced by an executable.

The machine learning model will learn patterns across multiple features
rather than relying on a single API.

---

# 🤖 Machine Learning

The project will investigate multiple machine learning algorithms.

### Initial Models

- Random Forest
- XGBoost
- LightGBM
- Support Vector Machine (SVM)

Additional algorithms may be investigated depending on experimental
requirements.

---

# 🔎 Unseen Ransomware Detection

A major component of this research is evaluation on ransomware families that
are not represented in the training data.

For example:

Training:

Family A
Family B
Family C
Family D

Testing:

Family E

Family E is intentionally excluded from the training set.

This allows the research to investigate how well the framework can identify
ransomware from a previously unseen family.

---

# 🚨 Novelty / Anomaly Detection

In addition to supervised classification, the research will investigate
novelty/anomaly detection techniques.

Candidate techniques include:

- Isolation Forest
- One-Class SVM

These methods will be investigated for identifying samples that differ
significantly from the learned distribution.

The final method will be selected based on experimental results.

---

# 📈 Evaluation Metrics

The models will be evaluated using multiple metrics rather than accuracy
alone.

Planned evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

For ransomware detection, particular attention will be given to recall,
precision and false-positive behavior.

---

# 🧠 Explainable AI

SHAP (SHapley Additive exPlanations) will be investigated to understand
which features contribute to model predictions.

The objective is to make model predictions more interpretable and identify
important features associated with ransomware classification.

---

# 🛠️ Technologies

## Programming Language

- Python 3.11.9

## Development Environment

- Visual Studio Code

## Version Control

- Git
- GitHub

## Data Processing

- NumPy
- Pandas

## Machine Learning

- Scikit-learn
- XGBoost
- LightGBM
- Imbalanced-learn

## Visualization

- Matplotlib
- Seaborn

## Explainable AI

- SHAP

## Research / Experimentation

- Jupyter Notebook

---

# 📁 Project Structure

```text
RP(AI)/
│
├── .venv/
│
├── data/
│   ├── raw/
│   │   └── RanDS_Static_API_Dataset.zip
│   │
│   ├── processed/
│   │
│   └── external/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_analysis.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_known_ransomware_detection.ipynb
│   ├── 06_unseen_family_detection.ipynb
│   └── 07_model_explainability.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_analysis.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   └── lightgbm_model.py
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── novelty_detection.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
├── docs/
│   ├── research_notes/
│   └── references/
│
├── .gitignore
├── README.md
└── requirements.txt