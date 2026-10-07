"""
Custom Scikit-Learn API Compliant Preprocessing Transformers
Implements leak-free Imputer, Scaler, OneHotEncoder, TargetEncoder, Pipeline, and ColumnTransformer.
"""

import numpy as np
import pandas as pd

class BaseTransformer:
    """Base class for all Scikit-Learn API compliant custom transformers."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class NumericalImputer(BaseTransformer):
    """Fills missing values in numerical columns using median/mean computed on training set."""
    def __init__(self, strategy='median', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            series = pd.to_numeric(X_df[col], errors='coerce')
            if self.strategy == 'median':
                val = series.median()
            elif self.strategy == 'mean':
                val = series.mean()
            elif self.strategy == 'constant':
                val = self.fill_value if self.fill_value is not None else 0.0
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            self.statistics_[col] = 0.0 if pd.isna(val) else float(val)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            series = pd.to_numeric(X_df[col], errors='coerce')
            fill_val = self.statistics_.get(col, 0.0)
            X_df[col] = series.fillna(fill_val)
        return X_df.values


class StandardScalerTransformer(BaseTransformer):
    """Standardizes features by removing mean and scaling to unit variance using training statistics."""
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        if self.with_mean:
            self.mean_ = np.nanmean(X_arr, axis=0)
        else:
            self.mean_ = np.zeros(X_arr.shape[1])
            
        if self.with_std:
            self.scale_ = np.nanstd(X_arr, axis=0)
            self.scale_[self.scale_ == 0.0] = 1.0  # Prevent division by zero
        else:
            self.scale_ = np.ones(X_arr.shape[1])
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.mean_) / self.scale_


class CategoricalImputer(BaseTransformer):
    """Fills missing or blank categorical values with a constant fill string."""
    def __init__(self, fill_value='Missing'):
        self.fill_value = fill_value

    def transform(self, X):
        X_df = pd.DataFrame(X).astype(str).copy()
        for col in X_df.columns:
            mask = X_df[col].str.strip().isin(['', 'nan', 'None', 'null', 'NaN', 'N/A'])
            X_df.loc[mask, col] = self.fill_value
        return X_df.values


class OneHotEncoderTransformer(BaseTransformer):
    """
    One-Hot Encoder with handle_unknown='ignore' support.
    Unseen categories during transform receive all 0s.
    """
    def __init__(self, handle_unknown='ignore', drop_first=False):
        self.handle_unknown = handle_unknown
        self.drop_first = drop_first
        self.categories_ = {}
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).astype(str)
        self.categories_ = {}
        self.feature_names_out_ = []
        
        for col in X_df.columns:
            unique_cats = sorted(X_df[col].unique().tolist())
            if self.drop_first and len(unique_cats) > 1:
                unique_cats = unique_cats[1:]
            self.categories_[col] = unique_cats
            for cat in unique_cats:
                self.feature_names_out_.append(f"{col}_{cat}")
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).astype(str)
        encoded_cols = []
        
        for col in X_df.columns:
            cats = self.categories_.get(col, [])
            for cat in cats:
                binary_series = (X_df[col] == cat).astype(float)
                encoded_cols.append(binary_series.values)
                
        if not encoded_cols:
            return np.empty((len(X_df), 0))
            
        return np.column_stack(encoded_cols)

    def get_feature_names_out(self):
        return self.feature_names_out_


class TargetEncoderTransformer(BaseTransformer):
    """
    Smoothed m-estimate Target Encoder for categorical features.
    S_i = (n_i * mean_i + m * global_mean) / (n_i + m)
    Fitted strictly on training target y to prevent data leakage.
    """
    def __init__(self, smoothing=10.0):
        self.smoothing = smoothing
        self.global_mean_ = 0.0
        self.target_maps_ = {}

    def fit(self, X, y):
        if y is None:
            raise ValueError("TargetEncoder requires target label y during fit().")
            
        X_df = pd.DataFrame(X).astype(str)
        y_arr = np.asarray(y, dtype=float)
        self.global_mean_ = float(np.mean(y_arr))
        self.target_maps_ = {}

        for col in X_df.columns:
            col_data = X_df[col]
            df_temp = pd.DataFrame({'cat': col_data, 'target': y_arr})
            stats = df_temp.groupby('cat')['target'].agg(['count', 'mean'])
            
            # Smoothed m-estimate
            counts = stats['count']
            means = stats['mean']
            smoothed = (counts * means + self.smoothing * self.global_mean_) / (counts + self.smoothing)
            self.target_maps_[col] = smoothed.to_dict()
            
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).astype(str).copy()
        encoded_cols = []
        
        for col in X_df.columns:
            map_dict = self.target_maps_.get(col, {})
            encoded = X_df[col].map(map_dict).fillna(self.global_mean_).values
            encoded_cols.append(encoded)
            
        return np.column_stack(encoded_cols)


class Pipeline:
    """Sequential execution of Scikit-Learn compliant transformer steps."""
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        X_trans = X
        for name, transformer in self.steps:
            if hasattr(transformer, 'fit'):
                if isinstance(transformer, TargetEncoderTransformer):
                    transformer.fit(X_trans, y)
                else:
                    transformer.fit(X_trans, y)
            if hasattr(transformer, 'transform'):
                X_trans = transformer.transform(X_trans)
        return self

    def transform(self, X):
        X_trans = X
        for name, transformer in self.steps:
            X_trans = transformer.transform(X_trans)
        return X_trans

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        last_name, last_trans = self.steps[-1]
        if hasattr(last_trans, 'get_feature_names_out'):
            return last_trans.get_feature_names_out()
        elif input_features is not None:
            return input_features
        else:
            return [f"feature_{i}" for i in range(100)]


class ColumnTransformer:
    """Applies separate preprocessing pipelines to subsets of DataFrame columns."""
    def __init__(self, transformers):
        self.transformers = transformers
        self.fitted_transformers_ = []
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.fitted_transformers_ = []
        self.feature_names_out_ = []

        for name, pipe, cols in self.transformers:
            # Subset input columns
            sub_X = X_df[cols].values if isinstance(cols, list) else X_df.iloc[:, cols].values
            
            # Clone/fit pipeline
            pipe.fit(sub_X, y)
            self.fitted_transformers_.append((name, pipe, cols))
            
            # Collect feature names out
            if hasattr(pipe, 'get_feature_names_out'):
                names = pipe.get_feature_names_out(cols)
            else:
                names = [f"{name}__{c}" for c in cols]
            self.feature_names_out_.extend(names)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        transformed_blocks = []

        for name, pipe, cols in self.fitted_transformers_:
            sub_X = X_df[cols].values if isinstance(cols, list) else X_df.iloc[:, cols].values
            sub_trans = pipe.transform(sub_X)
            transformed_blocks.append(sub_trans)

        return np.column_stack(transformed_blocks)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self):
        return self.feature_names_out_
