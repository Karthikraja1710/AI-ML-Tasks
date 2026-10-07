import json
import os

def generate_feature_engineering_notebook():
    cells = []

    # Cell 1: Notebook Title & Overview (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Feature Engineering & Preprocessing Pipeline\n",
            "\n",
            "**Project Title:** Feature Engineering & Preprocessing Pipeline  \n",
            "**Notebook Focus:** Reusable, Leak-Free Scikit-Learn Preprocessing Pipeline (`ColumnTransformer`, Imputation, Scaling, One-Hot Encoding, Target Encoding, Correlation Analysis, & Unseen Category Handling)  \n",
            "**Dataset:** Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)  \n",
            "\n",
            "---  \n",
            "## Key Pipeline Architecture Highlights\n",
            "1. **Leak-Free Preprocessing:** All scalers, imputers, and encoders fit strictly on `X_train` and `y_train`.\n",
            "2. **Stratified Splitting:** Preserves the 26.54% churn target ratio identically across training (80%) and testing (20%) folds.\n",
            "3. **Modular Sub-Pipelines:** Integrates numerical median imputation + Z-score scaling, categorical imputation + One-Hot Encoding (`handle_unknown='ignore'`), and Target Encoding.\n",
            "4. **Production Inference Resilience:** Automatically handles unseen categories without throwing errors or NaNs.\n"
        ]
    })

    # Cell 2: Imports & Environment Setup (Code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "\n",
            "# Include src directory in module path\n",
            "project_root = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()\n",
            "if project_root not in sys.path:\n",
            "    sys.path.append(project_root)\n",
            "\n",
            "from src.transformers import (\n",
            "    NumericalImputer, CategoricalImputer, StandardScalerTransformer,\n",
            "    OneHotEncoderTransformer, TargetEncoderTransformer, Pipeline, ColumnTransformer\n",
            ")\n",
            "from src.pipeline_builder import build_preprocessing_pipeline\n",
            "from src.correlation_analysis import analyze_numerical_correlations\n",
            "from src.leakage_checks import run_leakage_and_integrity_checks\n",
            "\n",
            "print('Environment initialized successfully. Pandas version:', pd.__version__)\n"
        ]
    })

    # Cell 3: Data Ingestion & Schema Inspection (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 1: Data Loading, Feature Schema Inspection, & Missingness Audit\n",
            "We load the dataset, identify feature data types, target labels, and inspect missing values."
        ]
    })

    # Cell 4: Data Ingestion Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "data_path = os.path.join(project_root, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')\n",
            "df = pd.read_csv(data_path)\n",
            "\n",
            "print('=== DATASET OVERVIEW ===')\n",
            "print(f'Total Rows: {len(df)}, Total Columns: {len(df.columns)}')\n",
            "print('\\nTarget Distribution (Churn):')\n",
            "print(df['Churn'].value_counts(normalize=True))\n",
            "\n",
            "# Audit Blank TotalCharges\n",
            "blank_tc = (df['TotalCharges'].astype(str).str.strip() == '').sum()\n",
            "print(f'\\nBlank TotalCharges Count: {blank_tc}')\n",
            "print('Tenure of Blank TotalCharges Rows:', df[df['TotalCharges'].astype(str).str.strip() == '']['tenure'].tolist())\n"
        ]
    })

    # Cell 5: Stratified Split Rationale (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 2: Stratified Train/Test Data Splitting\n",
            "\n",
            "### Splitting Strategy Justification\n",
            "* **Stratified 80/20 Split:** Preserves target class proportions (26.54% churn) identically in training ($N=5,636$) and test ($N=1,407$) sets.\n",
            "* **Why Not Temporal Split?** Dataset provides elapsed `tenure` but lacks snapshot calendar timestamps. Sorting by tenure would distort feature distributions across folds.\n"
        ]
    })

    # Cell 6: Stratified Splitting Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "y = (df['Churn'] == 'Yes').astype(int).values\n",
            "X = df.drop(columns=['customerID', 'Churn']).copy()\n",
            "\n",
            "num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']\n",
            "cat_cols = [c for c in X.columns if c not in num_cols]\n",
            "\n",
            "# Leak-free Stratified Split\n",
            "np.random.seed(42)\n",
            "indices = np.arange(len(df))\n",
            "pos_idx = indices[y == 1]\n",
            "neg_idx = indices[y == 0]\n",
            "\n",
            "np.random.shuffle(pos_idx)\n",
            "np.random.shuffle(neg_idx)\n",
            "\n",
            "n_pos_test = int(len(pos_idx) * 0.2)\n",
            "n_neg_test = int(len(neg_idx) * 0.2)\n",
            "\n",
            "test_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])\n",
            "train_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])\n",
            "\n",
            "X_train, y_train = X.iloc[train_idx].copy(), y[train_idx]\n",
            "X_test, y_test = X.iloc[test_idx].copy(), y[test_idx]\n",
            "\n",
            "print(f'Train Fold Shape: {X_train.shape}, Target Churn Proportion: {y_train.mean():.4f}')\n",
            "print(f'Test Fold Shape : {X_test.shape}, Target Churn Proportion: {y_test.mean():.4f}')\n"
        ]
    })

    # Cell 7: Building Pipeline Architecture (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 3: Scikit-Learn Pipeline & ColumnTransformer Construction\n",
            "\n",
            "We construct independent sub-pipelines for numerical scaling and categorical encoding, unified inside a `ColumnTransformer`."
        ]
    })

    # Cell 8: Pipeline Construction & Fitting Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "pipeline = build_preprocessing_pipeline(num_cols, cat_cols)\n",
            "\n",
            "# Fit strictly on training fold\n",
            "pipeline.fit(X_train, y_train)\n",
            "\n",
            "# Transform train and test sets\n",
            "X_train_trans = pipeline.transform(X_train)\n",
            "X_test_trans = pipeline.transform(X_test)\n",
            "\n",
            "print('Pipeline Fit & Transform Successful!')\n",
            "print(f'Transformed X_train Shape: {X_train_trans.shape}')\n",
            "print(f'Transformed X_test Shape : {X_test_trans.shape}')\n",
            "print(f'Engineered Features Count: {len(pipeline.get_feature_names_out())}')\n"
        ]
    })

    # Cell 9: Correlation Analysis (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 4: Numerical Feature Correlation Analysis\n",
            "\n",
            "We analyze Pearson ($r$) and Spearman ($\rho$) correlation matrices to detect multi-collinearity."
        ]
    })

    # Cell 10: Correlation Analysis Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "corr_results = analyze_numerical_correlations(X_train, num_cols)\n",
            "print('=== PEARSON CORRELATION MATRIX ===')\n",
            "print(corr_results['pearson'])\n",
            "\n",
            "print('\\n=== SPEARMAN CORRELATION MATRIX ===')\n",
            "print(corr_results['spearman'])\n",
            "\n",
            "print('\\nHighly Correlated Feature Pairs (|r| >= 0.70):')\n",
            "for pair in corr_results['high_corr_pairs']:\n",
            "    print(f\"  {pair['feature_1']} <-> {pair['feature_2']}: Pearson r = {pair['pearson_r']:.4f}, Spearman rho = {pair['spearman_rho']:.4f}\")\n"
        ]
    })

    # Cell 11: Production Resilience & Unseen Category Test (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 5: Inference Reusability & Unseen Category Resilience Test\n",
            "\n",
            "We test the pipeline on synthetic new customer records containing unknown category strings (e.g. `PaymentMethod = 'Crypto / Bitcoin'`)."
        ]
    })

    # Cell 12: Inference Test Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "sample_unseen = X_test.iloc[:3].copy()\n",
            "sample_unseen['PaymentMethod'] = ['Crypto / Bitcoin', 'Apple Pay', 'PayPal']\n",
            "sample_unseen['InternetService'] = ['Satellite', 'Quantum Fiber', 'Starlink']\n",
            "\n",
            "unseen_trans = pipeline.transform(sample_unseen)\n",
            "print('Unseen Category Inference Output Shape:', unseen_trans.shape)\n",
            "print('NaN Count in Inference Output:', np.isnan(unseen_trans).sum())\n",
            "print('Pipeline successfully transformed unseen categories with 0 errors!')\n"
        ]
    })

    # Cell 13: Automated Leakage Checks (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 6: Automated Data Leakage & Pipeline Integrity Assertions\n",
            "\n",
            "We execute automated assertions validating zero data leakage."
        ]
    })

    # Cell 14: Leakage Checks Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "checks = run_leakage_and_integrity_checks(pipeline, X_train, X_test, y_train, y_test, num_cols)\n",
            "print('=== AUTOMATED LEAKAGE & INTEGRITY CHECK RESULTS ===')\n",
            "for check_name, status in checks.items():\n",
            "    print(f'  {check_name:<35}: {status}')\n"
        ]
    })

    # Cell 15: Pipeline Summary & Conclusion (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 7: Summary of Pipeline Architecture & Dataset Limitations\n",
            "\n",
            "### Pipeline Architecture Summary\n",
            "1. **Leak-Free Guarantee:** Transformers are fitted strictly on `X_train`. Holdout test metrics remain unpolluted.\n",
            "2. **ColumnTransformer Integration:** Integrates median imputation, Z-score scaling, and One-Hot Encoding into a single executable object producing 46 dense output features.\n",
            "3. **Inference Resilience:** Handles unseen category strings gracefully (`handle_unknown='ignore'`).\n",
            "\n",
            "### Dataset Limitations\n",
            "* **Cross-Sectional Data:** Absence of historical temporal snapshots.\n",
            "* **High Multi-Collinearity:** Strong linear association between `tenure` and `TotalCharges` ($r = 0.8253$).\n"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    notebooks_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
    os.makedirs(notebooks_dir, exist_ok=True)
    output_path = os.path.join(notebooks_dir, 'feature_engineering_pipeline.ipynb')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Feature Engineering Notebook generated successfully at: {output_path}")

if __name__ == '__main__':
    generate_feature_engineering_notebook()
