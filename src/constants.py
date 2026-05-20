"""
Central constants for the BabiHost Price Predictor.
Consolidating these here avoids circular dependencies between modules.
"""

ROOM_TYPES = [
    'Entire home/apt',
    'Private room',
    'Shared room',
    'Hotel room'
]

# One-hot features are created manually to guarantee identical column order at training and inference
FEATURE_COLUMNS = [
    'bedrooms',
    'bathrooms',
    'accommodates',
    'room_type_entire',
    'room_type_private',
    'room_type_shared',
    'room_type_hotel',
    'neighbourhood_freq',
    'num_amenities',
    'availability_ratio',
    'is_superhost',
    'host_verified',
]
