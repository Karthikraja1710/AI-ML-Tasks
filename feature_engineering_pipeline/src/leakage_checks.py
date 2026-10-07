"""
Data Leakage Prevention Checks & Assertions Module
Validates pipeline transformation integrity, zero test leakage, and unseen category handling.
"""

import numpy as np
import pandas as pd

def run_leakage_and_integrity_checks(pipeline, X_train, X_test, y_train, y_test, num_cols):
    """
    Executes automated assertions to confirm zero data leakage and pipeline usability.
    
    Args:
        pipeline: Fitted ColumnTransformer or Pipeline instance.
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Test feature matrix.
        y_train (np.ndarray): Training target vector.
        y_test (np.ndarray): Test target vector.
        num_cols (list): Numerical column names.
        
    Returns:
        results (dict): Assertion status logs and metrics.
    """
    X_train_trans = pipeline.transform(X_train)
    X_test_trans = pipeline.transform(X_test)
    
    # Assertion 1: No NaN or Inf values in transformed outputs
    assert not np.isnan(X_train_trans).any(), "ASSERTION ERROR: NaN values found in X_train_trans!"
    assert not np.isnan(X_test_trans).any(), "ASSERTION ERROR: NaN values found in X_test_trans!"
    assert not np.isinf(X_train_trans).any(), "ASSERTION ERROR: Inf values found in X_train_trans!"
    assert not np.isinf(X_test_trans).any(), "ASSERTION ERROR: Inf values found in X_test_trans!"
    
    # Assertion 2: Training set numerical features have ~0 mean and ~1 std
    # Numerical features are the first len(num_cols) columns
    n_num = len(num_cols)
    train_num_means = np.mean(X_train_trans[:, :n_num], axis=0)
    train_num_stds = np.std(X_train_trans[:, :n_num], axis=0)
    
    assert np.allclose(train_num_means, 0.0, atol=1e-4), f"ASSERTION ERROR: Training numerical means not 0! {train_num_means}"
    assert np.allclose(train_num_stds, 1.0, atol=1e-4), f"ASSERTION ERROR: Training numerical stds not 1! {train_num_stds}"
    
    # Assertion 3: Test set numerical means are NOT exactly 0 (verifies scalers fit ONLY on train)
    test_num_means = np.mean(X_test_trans[:, :n_num], axis=0)
    assert not np.allclose(test_num_means, 0.0, atol=1e-4), "ASSERTION ERROR: Test numerical means are exactly 0! Possible data leakage."
    
    # Assertion 4: Unseen Category Resilience Test
    sample_inference = X_test.iloc[:2].copy()
    # Inject unknown unseen categories
    for col in sample_inference.columns:
        if sample_inference[col].dtype == object or col == 'PaymentMethod':
            sample_inference[col] = "UNSEEN_NEW_CATEGORY_999"
            
    try:
        inf_trans = pipeline.transform(sample_inference)
        assert inf_trans.shape[0] == 2, "ASSERTION ERROR: Inference output shape mismatch!"
        assert not np.isnan(inf_trans).any(), "ASSERTION ERROR: Inference output contains NaNs for unseen categories!"
        unseen_resilience = True
    except Exception as e:
        unseen_resilience = False
        raise AssertionError(f"ASSERTION ERROR: Pipeline failed on unseen categories: {e}")
        
    return {
        'no_nans_pass': True,
        'train_scaled_mean_zero_pass': True,
        'test_scaled_mean_diff_pass': True,
        'unseen_category_resilience_pass': unseen_resilience,
        'train_transformed_shape': X_train_trans.shape,
        'test_transformed_shape': X_test_trans.shape
    }
