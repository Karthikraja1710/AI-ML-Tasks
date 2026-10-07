"""
Baseline Models Module
Contains implementations for Majority Baseline, Domain Heuristic Baseline, and Numpy Logistic Regression.
"""

import numpy as np
import pandas as pd

class MajorityBaseline:
    """Predicts majority class (0 = 'No Churn') for all instances."""
    def fit(self, X, y):
        self.majority_class_ = 0
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)
        
    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 1.0
        return probs

class HeuristicBaseline:
    """Predicts Churn=1 if Contract == 'Month-to-month' and InternetService == 'Fiber optic'."""
    def fit(self, X_df, y):
        return self

    def predict(self, df_raw):
        mask = (df_raw['Contract'] == 'Month-to-month') & (df_raw['InternetService'] == 'Fiber optic')
        return mask.astype(int).values

    def predict_proba(self, df_raw):
        preds = self.predict(df_raw)
        probs = np.zeros((len(df_raw), 2))
        probs[:, 1] = np.where(preds == 1, 0.80, 0.10)
        probs[:, 0] = 1.0 - probs[:, 1]
        return probs

class NumpyLogisticRegression:
    """
    Standardized Logistic Regression fitted via Batch Gradient Descent.
    Enforces zero data leakage by computing mean and std strictly on training set.
    """
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.mean_ = None
        self.std_ = None
        self.weights_ = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

    def fit(self, X_train, y_train):
        # Scale features using training statistics ONLY
        self.mean_ = X_train.mean(axis=0)
        self.std_ = X_train.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # Avoid division by zero
        
        X_scaled = (X_train - self.mean_) / self.std_
        X_b = np.hstack([np.ones((len(X_scaled), 1)), X_scaled])
        
        n_samples, n_features = X_b.shape
        self.weights_ = np.zeros(n_features)
        
        for _ in range(self.epochs):
            preds = self._sigmoid(X_b @ self.weights_)
            error = preds - y_train
            gradient = (X_b.T @ error) / n_samples
            self.weights_ -= self.lr * gradient
            
        return self

    def predict_proba(self, X_test):
        X_scaled = (X_test - self.mean_) / self.std_
        X_b = np.hstack([np.ones((len(X_scaled), 1)), X_scaled])
        p1 = self._sigmoid(X_b @ self.weights_)
        probs = np.vstack([1.0 - p1, p1]).T
        return probs

    def predict(self, X_test, threshold=0.5):
        probs = self.predict_proba(X_test)[:, 1]
        return (probs >= threshold).astype(int)
