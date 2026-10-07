"""
Pipeline Builder Module for Feature Engineering & Preprocessing
Constructs a Scikit-Learn API compliant ColumnTransformer and Pipeline for Telco Churn tabular dataset.
"""

import numpy as np
import pandas as pd
from .transformers import (
    NumericalImputer, CategoricalImputer, StandardScalerTransformer,
    OneHotEncoderTransformer, TargetEncoderTransformer, Pipeline, ColumnTransformer
)

def build_preprocessing_pipeline(num_cols, cat_cols, target_encode_cols=None):
    """
    Constructs a ColumnTransformer integrating numerical and categorical sub-pipelines.
    
    Args:
        num_cols (list): Numerical feature column names.
        cat_cols (list): Categorical feature column names for One-Hot Encoding.
        target_encode_cols (list, optional): Categorical features for Target Encoding.
        
    Returns:
        ct (ColumnTransformer): Fitted or unfitted ColumnTransformer.
    """
    # 1. Numerical Sub-Pipeline: Median Imputation + Standard Scaling
    num_pipeline = Pipeline([
        ('imputer', NumericalImputer(strategy='median')),
        ('scaler', StandardScalerTransformer(with_mean=True, with_std=True))
    ])
    
    # 2. Categorical Sub-Pipeline: Constant Imputation + One-Hot Encoding
    cat_pipeline = Pipeline([
        ('imputer', CategoricalImputer(fill_value='Missing')),
        ('ohe', OneHotEncoderTransformer(handle_unknown='ignore'))
    ])
    
    transformers = [
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ]
    
    # 3. Optional High-Cardinality Target Encoding Pipeline
    if target_encode_cols:
        target_pipeline = Pipeline([
            ('imputer', CategoricalImputer(fill_value='Missing')),
            ('target_enc', TargetEncoderTransformer(smoothing=10.0))
        ])
        transformers.append(('target_enc', target_pipeline, target_encode_cols))
        
    ct = ColumnTransformer(transformers)
    return ct
