"""
Data Loading & Quality Audit Module
Provides reproducible dataset ingestion and quality inspection.
"""

import pandas as pd
import numpy as np

def load_and_audit_data(filepath: str):
    """
    Loads raw CSV dataset and performs empirical quality audit.
    
    Returns:
        df (pd.DataFrame): Processed DataFrame with clean TotalCharges.
        audit_summary (dict): Quality metrics and missingness counts.
    """
    df = pd.read_csv(filepath)
    
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Audit blank TotalCharges
    blank_total_charges = (df['TotalCharges'].astype(str).str.strip() == '').sum()
    blank_tenure = df[df['TotalCharges'].astype(str).str.strip() == '']['tenure'].tolist()
    
    # Clean TotalCharges without data leakage (strictly deterministic scalar replacement for 0 tenure)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)
    
    # Verify target label distribution
    churn_counts = df['Churn'].value_counts()
    churn_rate = (df['Churn'] == 'Yes').mean()
    
    audit_summary = {
        'total_rows': total_rows,
        'total_cols': total_cols,
        'blank_total_charges_count': int(blank_total_charges),
        'blank_total_charges_tenure': blank_tenure,
        'churn_counts': churn_counts.to_dict(),
        'churn_rate': float(churn_rate)
    }
    
    return df, audit_summary

if __name__ == '__main__':
    data_path = r"g:\New folder\RabTech\AI-ML-Tasks\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df, audit = load_and_audit_data(data_path)
    print("=== DATA AUDIT SUMMARY ===")
    print(f"Rows: {audit['total_rows']}, Cols: {audit['total_cols']}")
    print(f"Blank TotalCharges: {audit['blank_total_charges_count']} (Tenure: {audit['blank_total_charges_tenure']})")
    print(f"Churn Distribution: {audit['churn_counts']} (Rate: {audit['churn_rate']:.4f})")
