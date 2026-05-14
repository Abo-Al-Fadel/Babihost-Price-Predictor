"""
Data preprocessing utilities for the BabiHost Price Predictor.

Functions in this module mirror the cleaning steps performed in
notebook 01_EDA_and_cleaning so that the same logic can be reused
for new data or automated pipelines.
"""

import re
import numpy as np
import pandas as pd


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
        match = re.search(r'(\d+\.?\d*)', str(text))
        return float(match.group(1)) if match else np.nan

    return bathrooms_text_series.apply(_parse)


def filter_price_range(df, low=10, high=1000, price_col='price_clean'):
    """
    Remove outlier rows whose price falls outside [low, high].

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
    is_superhost, room_type_encoded, neighbourhood_freq
    """
    # Boolean mappings
    df['host_verified'] = df['host_identity_verified'].map({'t': 1, 'f': 0}).fillna(0)
    df['is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0}).fillna(0)

    # Amenity count (rough estimate via comma splitting)
    df['num_amenities'] = df['amenities'].str.count(',') + 1

    # Price per person (useful for EDA, not used as a model feature)
    df['price_per_person'] = df['price_clean'] / df['accommodates']

    # Availability ratio
    df['availability_ratio'] = df['availability_365'] / 365

    # Room type encoding
    from src.features import ROOM_TYPE_MAP
    df['room_type_encoded'] = df['room_type'].map(ROOM_TYPE_MAP)

    # Neighbourhood frequency encoding
    neigh_freq = df['neighbourhood_cleansed'].value_counts(normalize=True)
    df['neighbourhood_freq'] = df['neighbourhood_cleansed'].map(neigh_freq)

    return df
