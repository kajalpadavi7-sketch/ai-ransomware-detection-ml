# AI-Based Ransomware Detection Framework Using Machine Learning

## 📌 Project Overview

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
