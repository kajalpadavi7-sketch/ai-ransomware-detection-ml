# AI-Based Ransomware Detection Framework Using Machine Learning

## 📌 Project Overview

<<<<<<< HEAD
This research project focuses on developing an **AI-based ransomware detection
framework using machine learning and static analysis of Windows Portable
Executable (PE) files**.

The proposed framework is designed to classify PE files into:

- **Benign**
- **Ransomware**

The main focus of the research is to investigate whether static PE-based
features can be used to detect ransomware and generalize to ransomware
families that were not included during model training.

The framework will also investigate **novelty/anomaly detection** and
**Explainable AI (XAI)** to improve the interpretability of the detection
system.

> **Important:** This project is an academic research prototype. It does not
> claim guaranteed detection of genuine zero-day ransomware.

---

# 🎯 Research Objectives

The main objectives of this research are:

1. Develop a machine-learning-based ransomware detection framework.
2. Detect ransomware from benign PE files using static features.
3. Compare multiple machine learning algorithms.
4. Identify important static features associated with ransomware.
5. Evaluate the models using multiple cybersecurity-oriented metrics.
6. Evaluate generalization on ransomware families excluded from training.
7. Investigate novelty/anomaly detection for unfamiliar samples.
8. Use SHAP to explain machine-learning predictions.
9. Validate the selected model using an independent secondary dataset.
10. Develop a prototype application for PE-file-based ransomware prediction.

---

# 🔬 Research Problem

Traditional signature-based detection methods can struggle when malware is
modified or when a previously unseen ransomware variant appears.

Therefore, this research investigates a machine-learning approach based on
**static characteristics of Windows PE files**.

The research specifically evaluates whether a model trained on known
ransomware families can correctly classify ransomware samples from families
that were intentionally excluded from the training data.

---

# 🏗️ Proposed Research Framework

The proposed workflow is:

```text
RanDS Primary Dataset
        ↓
Dataset Understanding
        ↓
Data Preprocessing
        ↓
Feature Analysis
        ↓
Feature Selection
        ↓
Train / Validation / Test Split
        ↓
Class Imbalance Handling
        ↓
Machine Learning Models
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
Known Ransomware Detection
        ↓
Unseen-Family Evaluation
        ↓
Novelty / Anomaly Detection
        ↓
SHAP Explainability
        ↓
Secondary Dataset Validation
        ↓
Prototype Detection Application
=======
This is an academic research project for detecting ransomware in Windows Portable Executable (PE) files using **Machine Learning and static analysis**.

The **RanDS dataset** is used to train and evaluate multiple machine learning models. A major focus of the project is testing whether models trained on known ransomware families can detect ransomware from **unseen ransomware families**.

SHAP is also used for explainable AI to understand the importance of different features.

---

## 🎯 Objectives

- Detect ransomware using static PE features.
- Extract API and DLL-based statistical features.
- Train and compare multiple ML models.
- Evaluate performance on unseen ransomware families.
- Analyze feature importance using SHAP.
- Perform threshold analysis.
- Develop an `.exe` prediction prototype as future work.

---

## ⚙️ Features Used

```text
API_Count
DLL_Count
Unique_API_Count
Unique_DLL_Count
Size
Packed
Entropy
Year
```

---

## 🤖 Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

### 🏆 Best Current Model

**Random Forest**

| Metric | Result |
|---|---:|
| Accuracy | 90.20% |
| Precision | 98.88% |
| Recall | 87.02% |
| F1 Score | 92.57% |
| ROC-AUC | 97.72% |

---

## 📁 Project Structure

```text
ai-ransomware-detection-ml/
│
├── data/
│   ├── raw/
│   │   └── RanDS/
│   └── processed/
│       ├── benign_clean.csv
│       ├── ransomware_clean.csv
│       ├── api_features.csv
│       ├── ml_features.csv
│       ├── train_unseen_family.csv
│       └── test_unseen_family.csv
│
├── results/
│   ├── models/
│   │   ├── random_forest.joblib
│   │   ├── xgboost.joblib
│   │   ├── lightgbm.joblib
│   │   ├── decision_tree.joblib
│   │   └── logistic_regression.joblib
│   │
│   ├── evaluation/
│   │   ├── roc_curve.png
│   │   ├── precision_recall_curve.png
│   │   ├── threshold_analysis.csv
│   │   └── unseen_family_performance.csv
│   │
│   └── shap/
│       ├── feature_importance.csv
│       ├── shap_feature_importance.png
│       └── shap_summary_plot.png
│
├── src/
│   ├── preprocessing/
│   │   ├── clean_rands.py
│   │   └── prepare_features.py
│   ├── features/
│   │   └── extract_api_features.py
│   ├── models/
│   │   └── train_models.py
│   └── evaluation/
│       ├── create_unseen_family_split.py
│       ├── evaluate_models.py
│       ├── unseen_family_performance.py
│       ├── threshold_analysis.py
│       └── shap_analysis.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- SHAP
- Matplotlib
- Joblib
- Git & GitHub
- Visual Studio Code

---

## 🔄 Workflow

```text
RanDS Dataset
     ↓
Preprocessing
     ↓
API/DLL Feature Extraction
     ↓
ML Feature Preparation
     ↓
Unseen-Family Split
     ↓
Model Training
     ↓
Model Comparison
     ↓
Random Forest Selection
     ↓
Unseen-Family Evaluation
     ↓
Threshold Analysis
     ↓
SHAP Explainability
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/kajalpadavi7-sketch/ai-ransomware-detection-ml.git
cd ai-ransomware-detection-ml
```

Create virtual environment:

```powershell
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Run the scripts in order:

```powershell
python src/preprocessing/clean_rands.py
```

```powershell
python src/features/extract_api_features.py
```

```powershell
python src/preprocessing/prepare_features.py
```

```powershell
python src/evaluation/create_unseen_family_split.py
```

```powershell
python src/models/train_models.py
```

```powershell
python src/evaluation/evaluate_models.py
```

```powershell
python src/evaluation/unseen_family_performance.py
```

```powershell
python src/evaluation/threshold_analysis.py
```

```powershell
python src/evaluation/shap_analysis.py
```

---

## 🔬 Current Status

### Completed

- [x] RanDS preprocessing
- [x] API/DLL feature extraction
- [x] ML feature preparation
- [x] Unseen-family evaluation
- [x] Multiple ML models
- [x] Model comparison
- [x] Random Forest selection
- [x] Threshold analysis
- [x] SHAP explainability

### Future Work

- [ ] Robustness testing
- [ ] Unknown/anomaly detection
- [ ] Secondary dataset validation
- [ ] New `.exe` feature extraction
- [ ] `.exe` ransomware prediction prototype

---

## 👩‍💻 Author

**Kajal Padvi**  
MCA Research Project

> This project is intended for academic and defensive cybersecurity research. It does not guarantee zero-day ransomware detection.
==================================

1. Get overall metrics for the unseen-family test set
2. Investigate why Year is the dominant SHAP feature
3. Then proceed to novelty/anomaly detection
4. Then secondary dataset validation
5. Then the .exe prototype

========================

Phase 1 — Explainability

बाकी/अगला important काम

तुमने SHAP पहले किया है और तुम्हारे पास:

Year
Entropy
Size
Packed
Unique_API_Count
DLL_Count
API_Count
Unique_DLL_Count

की feature importance है।

लेकिन अब हमें इसे proper research experiment बनाना है:

Random Forest → SHAP → Global Feature Importance → Individual Prediction Explanation

और graphs/results save करने हैं।

Phase 2 — Unknown / Novel Ransomware Detection

यह तुम्हारे research का सबसे important novelty part है।

तुम्हारे paper का goal केवल:

"Ransomware vs Benign"

नहीं होना चाहिए।

बल्कि:

Known ransomware detection + unseen ransomware-family generalization + unknown/novel ransomware detection

हमें इसके लिए Isolation Forest / anomaly detection experiment properly करना होगा।

यही तुम्हारे project को simple ML classification project से research-oriented framework बनाने में मदद करेगा।

Phase 3 — Secondary Validation

अगर तुम्हारे research design में secondary dataset रखा है, तो उसके ऊपर model validation करना होगा।

उदाहरण:

Primary Dataset
      ↓
RanDS
      ↓
Training + Standard Test
      ↓
Unseen Family Test
      ↓
Unknown Detection
      ↓
Secondary Dataset Validation

यह research paper के लिए बहुत strong experiment होगा।

Phase 4 — Final Framework

इसके बाद पूरा framework:

PE File (.exe)
       ↓
Static Feature Extraction
       ↓
Preprocessing
       ↓
Feature Selection
       ↓
Random Forest
       ↓
Known / Ransomware Detection
       ↓
Unseen Family Evaluation
       ↓
Unknown / Novel Detection
       ↓
SHAP Explainability
       ↓
Final Prediction

को implement/document करना है।

Phase 5 — Final Graphs & Tables

Paper के लिए हमें final figures बनाने होंगे:

Model comparison
Accuracy comparison
Precision/Recall/F1 comparison
ROC curve
Precision-Recall curve
Confusion matrix
SHAP feature importance
Unseen-family recall distribution
Threshold vs F1
Threshold vs Recall/FNR
Known vs unseen performance comparison
Phase 6 — Research Paper

इसके बाद paper writing:

1. Abstract

2. Keywords

3. Introduction

4. Related Work / Literature Review

5. Research Gap

6. Proposed Methodology

7. Dataset Description

8. Feature Extraction

9. Machine Learning Models

10. Experimental Setup

11. Standard Evaluation Results

12. Unseen-Family Evaluation

13. Unknown/Novel Detection

14. Explainable AI using SHAP

15. Discussion

16. Limitations

17. Conclusion

18. Future Work

19. References

📊 मेरा current estimate

अगर सिर्फ ML implementation की बात करें:

लगभग 90% complete ✅

अगर पूरा research project + experiments देखें:

लगभग 75–80% complete ✅

अगर publication-ready research paper देखें:

लगभग 55–65% complete — क्योंकि experiments के results आ गए हैं, लेकिन methodology, novelty experiment, discussion, figures, tables और proper paper writing बाकी है।

सबसे important:

अभी नया model train करने की जरूरत नहीं है।
तुम्हारे पास standard evaluation + unseen-family evaluation + threshold analysis already है।

अब हमें SHAP → Unknown/Novel Detection → Secondary Validation → Final graphs/tables → Research Paper की तरफ जाना चाहिए।

और तुम्हारा generate_final_results.py successfully run हो चुका है, इसलिए final result aggregation भी complete है।

Next step मैं तुम्हें SHAP का proper research-level code दूँगा, फिर उसके बाद Unknown/Novel ransomware detection वाला module बनाएँगे।
