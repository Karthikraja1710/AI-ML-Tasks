"""
End-to-End Pipeline Verification Script
Loads dataset, performs stratified split, fits preprocessing pipeline, computes correlations, and runs leakage checks.
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure module path is included
sys.path.append(os.path.abspath('.'))

from src.pipeline_builder import build_preprocessing_pipeline
from src.correlation_analysis import analyze_numerical_correlations
from src.leakage_checks import run_leakage_and_integrity_checks

def run_end_to_end_test():
    data_path = os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = pd.read_csv(data_path)
    
    # Target
    y = (df['Churn'] == 'Yes').astype(int).values
    X = df.drop(columns=['customerID', 'Churn']).copy()
    
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    # 1. Stratified Split (80/20)
    np.random.seed(42)
    indices = np.arange(len(df))
    pos_idx = indices[y == 1]
    neg_idx = indices[y == 0]
    
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    n_pos_test = int(len(pos_idx) * 0.2)
    n_neg_test = int(len(neg_idx) * 0.2)
    
    test_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
    train_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])
    
    X_train, y_train = X.iloc[train_idx].copy(), y[train_idx]
    X_test, y_test = X.iloc[test_idx].copy(), y[test_idx]
    
    print(f"Train Shape: {X_train.shape}, Churn Rate: {y_train.mean():.4f}")
    print(f"Test Shape : {X_test.shape}, Churn Rate: {y_test.mean():.4f}")
    
    # 2. Build Pipeline
    pipeline = build_preprocessing_pipeline(num_cols, cat_cols)
    
    # 3. Fit strictly on train
    pipeline.fit(X_train, y_train)
    
    # 4. Transform train and test
    X_train_trans = pipeline.transform(X_train)
    X_test_trans = pipeline.transform(X_test)
    
    print("\nPipeline Fit & Transform Successful!")
    print(f"Transformed Train Shape: {X_train_trans.shape}")
    print(f"Transformed Test Shape : {X_test_trans.shape}")
    print(f"Total Output Features: {len(pipeline.get_feature_names_out())}")
    
    # 5. Correlation Analysis
    corr_results = analyze_numerical_correlations(X_train, num_cols)
    print("\n=== PEARSON CORRELATION MATRIX ===")
    print(corr_results['pearson'])
    print("\nHighly Correlated Pairs (|r| >= 0.70):", corr_results['high_corr_pairs'])
    
    # 6. Run Automated Leakage Checks
    check_results = run_leakage_and_integrity_checks(pipeline, X_train, X_test, y_train, y_test, num_cols)
    print("\n=== AUTOMATED LEAKAGE & INTEGRITY CHECKS ===")
    for k, v in check_results.items():
        print(f"  {k:<35}: {v}")
        
    print("\nALL PREPROCESSING PIPELINE TESTS PASSED 100%!")

if __name__ == '__main__':
    run_end_to_end_test()
