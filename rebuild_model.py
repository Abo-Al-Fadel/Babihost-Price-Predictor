"""
Automated rebuild script for the BabiHost Price Predictor model.
Cleans raw Cape Town listings.csv, runs the one-hot feature engineering pipeline,
tunes a Random Forest Regressor, evaluates it, and saves the production pickle files.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import clean_price, extract_bathrooms, filter_price_range, add_engineered_features
from src.train_model import prepare_xy, split_data, tune_rf, evaluate_model, save_model

def rebuild():
    print("[Model Rebuild] Starting model rebuild pipeline...")

    raw_data_path = os.path.join("data", "listings.csv")
    cleaned_data_path = os.path.join("data", "listings_cleaned.csv")

    if not os.path.exists(raw_data_path):
        print(f"[Model Rebuild] Error: Raw data not found at {raw_data_path}")
        return

    # 1. Loading & Cleaning (EDA Phase)
    print("[Model Rebuild] Loading raw Cape Town Airbnb listings...")
    df = pd.read_csv(raw_data_path)
    print(f"   Original dataset shape: {df.shape}")

    # Remove rows with null prices
    df = df.dropna(subset=['price'])
    
    # Clean Price
    df['price_clean'] = clean_price(df['price'])
    
    # Outlier Filter (ZAR currency bounds)
    low_bound = 100
    high_bound = 10000
    df = filter_price_range(df, low=low_bound, high=high_bound)
    print(f"   Shape after filtering prices between {low_bound} and {high_bound} ZAR: {df.shape}")

    # Parse Bathrooms
    df['bathrooms'] = extract_bathrooms(df['bathrooms_text'])

    # Drop nulls in critical features
    critical_cols = ['bedrooms', 'bathrooms', 'accommodates', 'room_type', 'neighbourhood_cleansed']
    df_clean = df.dropna(subset=critical_cols).copy()
    print(f"   Shape after dropping critical NAs: {df_clean.shape}")
    print(f"   Dropped {len(df) - len(df_clean)} rows containing NA values in critical features.")

    # Save cleaned data
    df_clean.to_csv(cleaned_data_path, index=False)
    print(f"[Model Rebuild] Cleaned data saved to {cleaned_data_path}")

    # 2. Feature Engineering
    print("[Model Rebuild] Running feature engineering...")
    df_feats = add_engineered_features(df_clean)
    
    # Save neighbourhood frequency mapping
    neigh_freq_map = df_feats['neighbourhood_cleansed'].value_counts(normalize=True).to_dict()
    neigh_freq_path = os.path.join("app", "neighbourhood_freq.pkl")
    joblib.dump(neigh_freq_map, neigh_freq_path)
    print(f"[Model Rebuild] Saved neighbourhood frequency mapping to {neigh_freq_path} ({len(neigh_freq_map)} wards)")

    # Save feature columns description (JSON/pickle)
    from src.constants import FEATURE_COLUMNS
    feature_cols_path = os.path.join("app", "feature_columns.pkl")
    joblib.dump(FEATURE_COLUMNS, feature_cols_path)
    print(f"[Model Rebuild] Saved feature column list to {feature_cols_path}")

    # 3. Model Training & Tuning
    print("[Model Rebuild] Preparing train/test splits...")
    X, y = prepare_xy(df_feats)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")

    print("[Model Rebuild] Running Random Forest hyperparameter tuning (RandomizedSearchCV)...")
    # Using small iter/cv for fast, lightweight yet highly effective tuning
    tuned_model, best_params = tune_rf(X_train, y_train, n_iter=20, cv=5, random_state=42)
    print(f"   Best Hyperparameters found: {best_params}")

    # 4. Evaluation
    print("[Model Rebuild] Evaluating model on unseen test data...")
    metrics = evaluate_model(tuned_model, X_test, y_test)
    print(f"   Metrics: MAE = {metrics['MAE']:.4f} ZAR, RMSE = {metrics['RMSE']:.4f} ZAR, R2 = {metrics['R2']:.4f}")

    # 5. Persist Model
    model_path = save_model(tuned_model, "model_tuned.pkl")
    print(f"[Model Rebuild] Production-ready tuned model saved to {model_path} ({os.path.getsize(model_path) / 1024 / 1024:.2f} MB)")
    print("[Model Rebuild] Model rebuild completed successfully!")

if __name__ == "__main__":
    rebuild()
