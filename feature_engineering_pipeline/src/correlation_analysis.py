"""
Correlation Analysis Module
Computes Pearson and Spearman correlation matrices for numerical features and provides collinearity insights.
"""

import numpy as np
import pandas as pd

def analyze_numerical_correlations(df, num_cols):
    """
    Calculates Pearson and Spearman correlation matrices for specified numerical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        num_cols (list): List of numerical column names.
        
    Returns:
        dict: Containing Pearson and Spearman correlation DataFrames and high-correlation pairs.
    """
    df_num = df[num_cols].copy()
    for col in num_cols:
        df_num[col] = pd.to_numeric(df_num[col].astype(str).str.strip(), errors='coerce')
        
    pearson_corr = df_num.corr(method='pearson')
    spearman_corr = df_num.corr(method='spearman')
    
    # Identify highly correlated pairs (|r| > 0.70)
    high_corr_pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            col1 = num_cols[i]
            col2 = num_cols[j]
            r_val = pearson_corr.loc[col1, col2]
            s_val = spearman_corr.loc[col1, col2]
            if abs(r_val) >= 0.70 or abs(s_val) >= 0.70:
                high_corr_pairs.append({
                    'feature_1': col1,
                    'feature_2': col2,
                    'pearson_r': float(r_val),
                    'spearman_rho': float(s_val)
                })
                
    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'high_corr_pairs': high_corr_pairs
    }
