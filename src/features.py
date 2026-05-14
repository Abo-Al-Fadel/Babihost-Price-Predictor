"""
Feature engineering utilities for the BabiHost Price Predictor.

This module centralises the feature transformation logic used by
the Flask API, Streamlit dashboard, and Jupyter notebooks so that
any changes only need to be made in one place.
"""

import os
import joblib

# ── Constants ──────────────────────────────────────────────────────────────────

ROOM_TYPE_MAP = {
    'Entire home/apt': 2,
    'Private room': 1,
    'Shared room': 0,
    'Hotel room': 3,
}

FEATURE_COLUMNS = [
    'bedrooms', 'bathrooms', 'accommodates', 'room_type_encoded',
    'neighbourhood_freq', 'num_amenities', 'availability_ratio',
    'is_superhost', 'host_verified',
]

# ── Artifact helpers ───────────────────────────────────────────────────────────

def _artifacts_dir():
    """Return the absolute path to the app/ folder that stores .pkl files."""
    return os.path.join(os.path.dirname(__file__), '..', 'app')


def load_model():
    """Load and return the tuned Random Forest model."""
    path = os.path.join(_artifacts_dir(), 'model_tuned.pkl')
    return joblib.load(path)


def load_neighbourhood_freq():
    """Load and return the neighbourhood frequency dictionary."""
    path = os.path.join(_artifacts_dir(), 'neighbourhood_freq.pkl')
    return joblib.load(path)


# ── Feature transformation ────────────────────────────────────────────────────

def build_feature_vector(
    bedrooms,
    bathrooms,
    accommodates,
    room_type,
    neighbourhood,
    num_amenities,
    availability_365,
    is_superhost,
    host_verified,
    neigh_freq_map=None,
):
    """
    Build the 9-element feature vector expected by the trained model.

    Parameters
    ----------
    bedrooms : int
    bathrooms : float
    accommodates : int
    room_type : str
        One of the keys in ROOM_TYPE_MAP.
    neighbourhood : str
        Ward name used to look up frequency encoding.
    num_amenities : int
    availability_365 : int
        Number of available days in a year (0-365).
    is_superhost : bool
    host_verified : bool
    neigh_freq_map : dict or None
        Pre-loaded neighbourhood frequency map. If None, it will be loaded
        from disk automatically.

    Returns
    -------
    list[float]
        Feature vector in the order expected by the model.
    """
    if neigh_freq_map is None:
        neigh_freq_map = load_neighbourhood_freq()

    room_encoded = ROOM_TYPE_MAP.get(room_type, 1)
    neighbourhood_freq = neigh_freq_map.get(neighbourhood, 0.01)
    availability_ratio = availability_365 / 365.0

    return [
        bedrooms,
        bathrooms,
        accommodates,
        room_encoded,
        neighbourhood_freq,
        num_amenities,
        availability_ratio,
        1 if is_superhost else 0,
        1 if host_verified else 0,
    ]


def predict_price(model, features):
    """
    Return a rounded price prediction from a feature vector.

    Parameters
    ----------
    model : sklearn estimator
        The trained model (e.g. RandomForestRegressor).
    features : list[float]
        Feature vector produced by build_feature_vector().

    Returns
    -------
    float
        Predicted nightly price in USD, rounded to 2 decimals.
    """
    prediction = model.predict([features])[0]
    return round(prediction, 2)
