"""
Data preprocessing utilities for the BabiHost Price Predictor.

Functions in this module mirror the cleaning steps performed in
notebook 01_EDA_and_cleaning so that the same logic can be reused
for new data or automated pipelines.
"""

import re
import numpy as np
import pandas as pd

from src.constants import FEATURE_COLUMNS, ROOM_TYPES


def clean_price(price_series):
    """
    Convert a pandas Series of price strings ('$1,234.00') to float.

    Parameters
    ----------
    price_series : pd.Series
        Raw price column containing dollar signs and commas.

    Returns
    -------
    pd.Series
        Numeric price values as float.
    """
    return (
        price_series
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .astype(float)
    )


def extract_bathrooms(bathrooms_text_series):
    """
    Extract the numeric bathroom count from the bathrooms_text column.

    Parameters
    ----------
    bathrooms_text_series : pd.Series
        The 'bathrooms_text' column (e.g. '2 baths', '1 shared bath').

    Returns
    -------
    pd.Series
        Numeric bathroom count as float, NaN where parsing fails.
    """
    def _parse(text):
        if pd.isna(text):
            return np.nan
        # Find numbers like 1, 1.5, 2 etc. before 'bath' (case-insensitive)
        match = re.search(r'(\d+(?:\.\d+)?)\s*bath', str(text).lower())
        return float(match.group(1)) if match else np.nan

    return bathrooms_text_series.apply(_parse)


def filter_price_range(df, low=100, high=10000, price_col='price_clean'):
    """
    Remove outlier rows whose price falls outside [low, high].

    Note: The Cape Town Airbnb dataset lists prices in local South African Rand (ZAR)
    using the dollar sign. The previous bounds (10, 1000) filtered out 83% of the data,
    introducing significant selection bias. Shifting the limits to (100, 10000) ZAR
    retains a representative and standard segment of Cape Town accommodation.

    Parameters
    ----------
    df : pd.DataFrame
    low : float
    high : float
    price_col : str

    Returns
    -------
    pd.DataFrame
        Filtered copy of the dataframe.
    """
    mask = (df[price_col] >= low) & (df[price_col] <= high)
    return df.loc[mask].copy()


def add_engineered_features(df):
    """
    Add all derived feature columns used by the model.

    Modifies *df* in place and returns it for convenience.

    Columns added
    -------------
    host_verified, num_amenities, price_per_person, availability_ratio,
    is_superhost, room_type_entire, room_type_private, room_type_shared,
    room_type_hotel, neighbourhood_freq
    """
    # Boolean mappings
    df['host_verified'] = df['host_identity_verified'].map({'t': 1, 'f': 0}).fillna(0)
    df['is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0}).fillna(0)

    # Amenity count (rough estimate via comma splitting)
    df['num_amenities'] = df['amenities'].str.count(',') + 1

    # Price per person (useful for EDA, not used as a model feature)
    # Protection against divide-by-zero if accommodates is 0: replace with NaN
    df['price_per_person'] = df['price_clean'] / df['accommodates'].replace(0, np.nan)

    # Availability ratio
    df['availability_ratio'] = df['availability_365'] / 365.0

    # One-hot encoding for room types to avoid modeling categories as linear ordinal values
    df['room_type_entire'] = (df['room_type'] == 'Entire home/apt').astype(int)
    df['room_type_private'] = (df['room_type'] == 'Private room').astype(int)
    df['room_type_shared'] = (df['room_type'] == 'Shared room').astype(int)
    df['room_type_hotel'] = (df['room_type'] == 'Hotel room').astype(int)

    # Neighbourhood frequency encoding
    neigh_freq = df['neighbourhood_cleansed'].value_counts(normalize=True)
    df['neighbourhood_freq'] = df['neighbourhood_cleansed'].map(neigh_freq)

    return df
