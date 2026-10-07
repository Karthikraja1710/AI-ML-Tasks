# ML Problem Framing & Responsible Data Card Project

**Project Title:** ML Problem Framing & Responsible Data Card  
**Domain:** Telecommunications Customer Churn & Retention Intervention  
**Target Repository:** `AI-ML-Tasks`  
**Author:** AI/ML Systems Engineering  

---

## Executive Overview

This repository delivers a production-grade, one-day project focusing on **Machine Learning Problem Framing, Responsible AI Governance, Data Card Documentation, Operational Risk Management, and Financial Loss Optimization**.

Using the Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), this project rigorously evaluates whether machine learning is justified over simple non-ML baselines, quantifies the extreme cost asymmetry between False Positives ($C_{FP} = \$20$) and False Negatives ($C_{FN} = \$500$), and establishes a leak-free reproducible baseline pipeline.

> [!WARNING]
> **Proportional Claims & Governance Disclaimer**  
> Based on empirical analysis of this cross-sectional dataset, the baseline model achieves **86.86% Recall** at a tuned decision threshold ($p^* = 0.20$), reducing financial loss by **83.0%** compared to the majority-class baseline.  
> **This model is NOT claimed to be fully fair, safe, or deployment-ready without ongoing production drift monitoring, subgroup fairness audits, and automated safety guards.**

---

## Summary of Project Deliverables

| Deliverable # | Artifact Path | Core Content & Key Finding |
| :---: | :--- | :--- |
| **1** | [`docs/01_ml_problem_framing_memo.md`](file:///g:/New%20folder/RabTech/AI-ML-Tasks/docs/01_ml_problem_framing_memo.md) | **ML Problem Framing Memo:** Defines business decision, $30$-day prediction target, unit of observation, prediction/action windows, intended users, decision workflow, and non-ML baseline assessment. |
| **2** | [`docs/02_responsible_data_card.md`](file:///g:/New%20folder/RabTech/AI-ML-Tasks/docs/02_responsible_data_card.md) | **Responsible Data Card:** Documenting data provenance, collection & consent (marking unknowns), 21-attribute schema, sensitive fields (`gender`, `SeniorCitizen`), missingness (11 blank `TotalCharges`), and prohibited uses. |
| **3** | [`docs/03_risk_register.md`](file:///g:/New%20folder/RabTech/AI-ML-Tasks/docs/03_risk_register.md) | **Operational & Algorithmic Risk Register:** Matrix and table detailing 6 key risks (leakage, demographic bias, concept drift, spam fatigue, FN revenue loss, technical failure) with owners and unresolved questions. |
| **4** | [`docs/04_business_and_model_metrics.md`](file:///g:/New%20folder/RabTech/AI-ML-Tasks/docs/04_business_and_model_metrics.md) | **Business & Model Metrics Alignment:** 3-tier metric hierarchy, explicit FP vs FN cost formulation ($25:1$ ratio), mathematical threshold derivation ($p^* = 0.0385$), and operational threshold selection ($p^* = 0.20$). |
| **5** | [`notebooks/reproducible_baseline.ipynb`](file:///g:/New%20folder/RabTech/AI-ML-Tasks/notebooks/reproducible_baseline.ipynb) | **Reproducible Baseline Notebook:** Fully executed Jupyter Notebook implementing leak-free stratified splitting, non-ML baselines, standardized logistic regression, and cost-weighted threshold tuning. |
| **6** | [`src/`](file:///g:/New%20folder/RabTech/AI-ML-Tasks/src/) | **Modular Python Pipeline:** Modular Python code containing `data_loader.py`, `baseline_model.py`, `evaluation.py`, `create_notebook.py`, and `execute_notebook.py`. |

---

## Summary of Baseline Model Performance & Financial Impact

All evaluations performed on a leak-free 20% holdout test set ($N = 1,409$ accounts):

```
+---------------------------------------------------------------------------------------------+
| Model / Strategy                 | Accuracy | Precision | Recall | F1-Score | Financial Loss |
+----------------------------------+----------+-----------+--------+----------+----------------+
| Baseline 0: Majority Class       |  73.49%  |   0.00%   |  0.00% |  0.0000  |  $186,500.00   |
| Baseline 1: Domain Heuristic     |  76.47%  |  54.98%   | 62.20% |  0.5836  |   $74,300.00   |
| Baseline 2: LogReg (p = 0.50)    |  80.60%  |  67.12%   | 52.55% |  0.5895  |   $90,420.00   |
| Baseline 2: LogReg (Tuned p=0.20)|  71.22%  |  47.65%   | 86.86% |  0.6154  |   $31,620.00   |
+---------------------------------------------------------------------------------------------+
```

---

## Established Data Facts vs. Documented Unknowns

### Established Data Facts
1. Dataset contains 7,043 customer records across 21 attributes.
2. Overall target `Churn` proportion is 26.54% (1,869 churners, 5,174 non-churners).
3. `TotalCharges` contains exactly 11 whitespace blank values, strictly corresponding to `tenure == 0` (new accounts). Zero missing values exist elsewhere.
4. Senior citizens (`SeniorCitizen = 1`) display a 41.68% churn rate compared to 23.61% for non-seniors.

### Documented Unknowns & Assumptions
* **Unknown Provider & Region:** Parent telecom operator, country, and state are unstated in raw source files.
* **Unknown Temporal Epoch:** Dataset lacks calendar dates/timestamps. `tenure` is provided in months, but the snapshot calendar period is unknown.
* **Consent & Collection Context:** Opt-in secondary consent for machine learning retention modeling is unverified in raw files.

---

## Directory Structure

```
AI-ML-Tasks/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw Telco Churn Dataset
├── docs/
│   ├── 01_ml_problem_framing_memo.md         # ML Problem Framing Memo
│   ├── 02_responsible_data_card.md           # Responsible Data Card
│   ├── 03_risk_register.md                   # Operational Risk Register
│   └── 04_business_and_model_metrics.md      # Business & Model Metrics Guide
├── notebooks/
│   └── reproducible_baseline.ipynb           # Fully Executed Baseline Notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                         # Data Loading & Quality Audit Module
│   ├── baseline_model.py                      # Non-ML & ML Baseline Implementations
│   ├── evaluation.py                          # Metrics & Cost Calculation Functions
│   ├── create_notebook.py                     # Notebook Generation Script
│   └── execute_notebook.py                    # Programmatic Notebook Executor
├── README.md                                 # Project Overview & Execution Guide
└── requirements.txt                          # Python Package Dependencies
```

---

## Setup & Run Instructions

### Prerequisites
* Python 3.8+
* `pandas`, `numpy`, `jupyter` (standard library + core data science packages)

### 1. Installation
Clone the repository and install the lightweight requirements:
```bash
git clone https://github.com/Karthikraja1710/AI-ML-Tasks.git
cd AI-ML-Tasks
pip install -r requirements.txt
```

### 2. Running Data Loader & Baseline Evaluation Script
Run the standalone evaluation pipeline:
```bash
python src/test_baseline.py
```

### 3. Regenerating and Executing the Baseline Jupyter Notebook
To programmatically regenerate and re-run all notebook cells:
```bash
python src/create_notebook.py
python src/execute_notebook.py
```

### 4. Interactive Jupyter Execution
Launch Jupyter Notebook to interactively view or edit `reproducible_baseline.ipynb`:
```bash
jupyter notebook notebooks/reproducible_baseline.ipynb
```
