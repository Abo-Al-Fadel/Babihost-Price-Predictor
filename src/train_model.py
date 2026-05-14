"""
Model training utilities for the BabiHost Price Predictor.

Provides helper functions for training, evaluating, and persisting
scikit-learn regression models. These mirror the steps performed in
notebooks 02_feature_engineering and 03_model_tuning.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import FEATURE_COLUMNS


def prepare_xy(df, feature_cols=None, target_col='price_clean'):
    """
    Split a dataframe into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str] or None
        Defaults to FEATURE_COLUMNS.
    target_col : str

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
    return df[feature_cols], df[target_col]


def split_data(X, y, test_size=0.2, random_state=42):
    """Convenience wrapper around train_test_split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a fitted model and return a dict of metrics.

    Returns
    -------
    dict with keys: MAE, RMSE, R2
    """
    y_pred = model.predict(X_test)
    return {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
    }


def train_default_rf(X_train, y_train, n_estimators=100, random_state=42):
    """Train and return a default Random Forest regressor."""
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def tune_rf(X_train, y_train, n_iter=50, cv=5, random_state=42):
    """
    Perform RandomizedSearchCV on a Random Forest and return the best model.
    """
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=random_state),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_absolute_error',
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def save_model(model, filename='model_tuned.pkl'):
    """Save the model to the app/ directory."""
    path = os.path.join(os.path.dirname(__file__), '..', 'app', filename)
    joblib.dump(model, path)
    return path
