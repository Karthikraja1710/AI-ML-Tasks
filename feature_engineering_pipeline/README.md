# Feature Engineering & Preprocessing Pipeline Project

**Project Title:** Feature Engineering & Preprocessing Pipeline  
**Domain:** Tabular Machine Learning / Production Preprocessing Architecture  
**Target Artifact:** Leak-Free Scikit-Learn Pipeline & Executable Jupyter Notebook  
**Dataset:** Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)  

---

## Executive Overview

This repository delivers a focused, production-ready **Feature Engineering & Preprocessing Pipeline** project. Designed using a complex tabular dataset containing numerical features, ordinal attributes, nominal categories, and missing values, this pipeline provides a leak-free architecture using `ColumnTransformer` and custom Scikit-Learn API compliant transformers (`Pipeline`, `StandardScaler`, `NumericalImputer`, `CategoricalImputer`, `OneHotEncoder`, and `TargetEncoder`).

---

## Summary of Project Deliverables

| Deliverable # | Artifact Path | Core Content & Key Finding |
| :---: | :--- | :--- |
| **1** | [`notebooks/feature_engineering_pipeline.ipynb`](file:///g:/New%20folder/RabTech/AI-ML-Tasks/feature_engineering_pipeline/notebooks/feature_engineering_pipeline.ipynb) | **Executable Jupyter Notebook:** Fully executed notebook demonstrating schema inspection, stratified splitting, ColumnTransformer construction, correlation analysis, unseen category inference test, and automated leakage assertions. |
| **2** | [`docs/pipeline_design_and_data_card.md`](file:///g:/New%20folder/RabTech/AI-ML-Tasks/feature_engineering_pipeline/docs/pipeline_design_and_data_card.md) | **Pipeline Architecture & Data Card:** Comprehensive documentation covering dataset provenance, usage terms, download steps, target label, feature types, missingness, stratified split rationale, pipeline design, correlation analysis, and dataset limitations. |
| **3** | [`src/transformers.py`](file:///g:/New%20folder/RabTech/AI-ML-Tasks/feature_engineering_pipeline/src/transformers.py) | **Custom Scikit-Learn Compliant Transformers:** Standardized `BaseTransformer`, `NumericalImputer`, `StandardScalerTransformer`, `CategoricalImputer`, `OneHotEncoderTransformer` (`handle_unknown='ignore'`), `TargetEncoderTransformer` (smoothed m-estimate), `Pipeline`, and `ColumnTransformer`. |
| **4** | [`src/pipeline_builder.py`](file:///g:/New%20folder/RabTech/AI-ML-Tasks/feature_engineering_pipeline/src/pipeline_builder.py) | **Pipeline Factory:** Assembles numerical, categorical, and target-encoding sub-pipelines into a single `ColumnTransformer`. |
| **5** | [`src/correlation_analysis.py`](file:///g:/New%20folder/RabTech/AI-ML-Tasks/feature_engineering_pipeline/src/correlation_analysis.py) | **Correlation Analysis Engine:** Calculates Pearson and Spearman correlation matrices and identifies high collinearity pairs ($r \ge 0.70$). |
| **6** | [`src/leakage_checks.py`](file:///g:/New%20folder/RabTech/AI-ML-Tasks/feature_engineering_pipeline/src/leakage_checks.py) | **Automated Leakage Prevention Assertions:** Automated tests verifying zero NaNs, training unit variance, test set non-zero means, and unseen category inference resilience. |

---

## Key Pipeline Features & Guarantees

```
+------------------------------------------------------------------------------------------------+
| Key Pipeline Feature              | Implementation & Guarantee                                 |
+-----------------------------------+------------------------------------------------------------+
| 1. Zero Data Leakage              | Scalers, imputers, & target maps fit strictly on X_train.  |
| 2. Stratified Train/Test Split    | Preserves 26.54% target churn ratio identically (80/20).   |
| 3. Sub-Pipeline Integration       | Median Imputation + Z-score scaling + One-Hot Encoding.    |
| 4. Target Encoding Protection     | Smoothed m-estimate (alpha=10) fitted strictly on y_train. |
| 5. Unseen Category Resilience     | One-Hot Encoder sets unknown categories to all 0s.         |
| 6. Automated Integrity Assertions | Verified mean=0, std=1 on train, non-zero mean on test.    |
+------------------------------------------------------------------------------------------------+
```

---

## Correlation Analysis Summary

* **Highly Correlated Feature Pair:** `tenure` and `TotalCharges` exhibit strong linear correlation ($r = 0.8253$, Spearman $\rho = 0.8879$).
* **Handling Strategy:** Retain both features for tree-based models (GBDT / Random Forest); drop or create a ratio feature $\text{AvgMonthlyRate} = \frac{\text{TotalCharges}}{\text{tenure} + 1}$ for linear models.

---

## Setup & Run Instructions

### 1. Execute End-to-End Verification
```powershell
python src/test_pipeline.py
```

### 2. Regenerate & Execute Notebook Programmatically
```powershell
python src/create_notebook.py
python src/execute_notebook.py
```

### 3. Open Interactive Jupyter Notebook
```powershell
jupyter notebook notebooks/feature_engineering_pipeline.ipynb
```
