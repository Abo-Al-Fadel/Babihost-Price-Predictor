"""
Feature engineering and inference utilities for the BabiHost Price Predictor.

This module centralises the feature transformation and inference logic
used by the Flask API, Streamlit dashboard, and Jupyter notebooks.
"""

import os
import joblib
import numpy as np
import pandas as pd

from src.constants import FEATURE_COLUMNS, ROOM_TYPES

# ── Artifact helpers ───────────────────────────────────────────────────────────

def _artifacts_dir():
    """Return the absolute path to the app/ folder that stores .pkl files."""
    return os.path.join(os.path.dirname(__file__), '..', 'app')


def load_model():
    """Load and return the tuned Random Forest model."""
    path = os.path.join(_artifacts_dir(), 'model_tuned.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Please run training first.")
    return joblib.load(path)


def load_neighbourhood_freq():
    """Load and return the neighbourhood frequency dictionary."""
    path = os.path.join(_artifacts_dir(), 'neighbourhood_freq.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Neighbourhood frequency file not found at {path}.")
    return joblib.load(path)


# ── Feature transformation ────────────────────────────────────────────────────

def build_feature_vector(
    bedrooms,
    bathrooms,
    accommodates,
    room_type,
    neighbourhood,
    availability_365,
    is_superhost,
    host_verified,
    num_amenities=None,
    amenities=None,
    neigh_freq_map=None,
):
    """
    Build the 12-element feature list expected by the trained model.
    Applies strict bounding/clamping to protect against OOD inference values.

    Parameters
    ----------
    bedrooms : int or float
    bathrooms : float
    accommodates : int
    room_type : str
        One of ROOM_TYPES.
    neighbourhood : str
        Ward name used to look up frequency encoding.
    availability_365 : int or float
        Number of available days in a year (0-365).
    is_superhost : bool
    host_verified : bool
    num_amenities : int or None
        Pre-computed amenity count.
    amenities : list[str], str, or None
        Raw list of amenities or comma-separated amenities string. If provided,
        this will be used to compute the amenity count.
    neigh_freq_map : dict or None
        Pre-loaded neighbourhood frequency map. If None, it will be loaded.

    Returns
    -------
    list[float]
        Feature vector in the order expected by the model.
    """
    if neigh_freq_map is None:
        neigh_freq_map = load_neighbourhood_freq()

    # 1. Bounds Clamping & Validations (M6)
    bedrooms_clamped = float(np.clip(bedrooms if bedrooms is not None else 1, 0, 10))
    bathrooms_clamped = float(np.clip(bathrooms if bathrooms is not None else 1.0, 0.5, 10.0))
    accommodates_clamped = int(np.clip(accommodates if accommodates is not None else 2, 1, 20))
    availability_clamped = float(np.clip(availability_365 if availability_365 is not None else 200, 0, 365))
    availability_ratio = availability_clamped / 365.0

    # 2. Amenity Reconciliation (C3)
    if amenities is not None:
        if isinstance(amenities, list):
            amenity_count = len(amenities)
        elif isinstance(amenities, str):
            # Mirror the training string comma-count behavior
            amenity_count = amenities.count(',') + 1
        else:
            amenity_count = int(num_amenities) if num_amenities is not None else 10
    else:
        amenity_count = int(num_amenities) if num_amenities is not None else 10
    
    amenity_count_clamped = int(np.clip(amenity_count, 0, 100))

    # 3. One-hot Room Type encoding (M5)
    # Whitelist check
    room_type_clean = room_type if room_type in ROOM_TYPES else 'Private room'
    room_type_entire = 1 if room_type_clean == 'Entire home/apt' else 0
    room_type_private = 1 if room_type_clean == 'Private room' else 0
    room_type_shared = 1 if room_type_clean == 'Shared room' else 0
    room_type_hotel = 1 if room_type_clean == 'Hotel room' else 0

    # 4. Neighbourhood frequency lookup
    neighbourhood_freq = neigh_freq_map.get(neighbourhood, 0.01)

    return [
        bedrooms_clamped,
        bathrooms_clamped,
        accommodates_clamped,
        room_type_entire,
        room_type_private,
        room_type_shared,
        room_type_hotel,
        neighbourhood_freq,
        amenity_count_clamped,
        availability_ratio,
        1 if is_superhost else 0,
        1 if host_verified else 0,
    ]


def predict_price(model, features):
    """
    Return price prediction along with a 95% prediction interval (uncertainty range).

    Uncertainty is derived dynamically from the individual tree estimator variance
    within the Random Forest ensemble (M1).

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestRegressor
        The trained model.
    features : list[float]
        Feature vector produced by build_feature_vector().

    Returns
    -------
    dict
        Dictionary containing predicted_price, lower_bound, and upper_bound in USD (or ZAR/local price).
    """
    # Predict using a DataFrame with correct feature names to avoid sklearn UserWarnings (m4)
    df_feats = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    
    prediction = model.predict(df_feats)[0]
    
    # Calculate confidence interval from tree prediction variance (M1)
    if hasattr(model, "estimators_") and len(model.estimators_) > 1:
        tree_preds = [tree.predict(df_feats.values)[0] for tree in model.estimators_]
        std_error = np.std(tree_preds)
        # 95% Prediction Interval (1.96 * std_error)
        lower_bound = max(0.0, prediction - 1.96 * std_error)
        upper_bound = prediction + 1.96 * std_error
    else:
        # Fallback to model MAE as standard error proxy (approx. 122.0 ZAR/USD)
        lower_bound = max(0.0, prediction - 1.96 * 122.0)
        upper_bound = prediction + 1.96 * 122.0

    return {
        'predicted_price': round(float(prediction), 2),
        'lower_bound': round(float(lower_bound), 2),
        'upper_bound': round(float(upper_bound), 2)
    }
