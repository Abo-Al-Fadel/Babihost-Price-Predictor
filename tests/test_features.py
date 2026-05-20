import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.constants import FEATURE_COLUMNS
from src.features import build_feature_vector, load_neighbourhood_freq

@pytest.fixture
def dummy_freq_map():
    return {
        "Ward 115": 0.05,
        "Camps Bay": 0.02,
    }

def test_feature_vector_shape_and_defaults(dummy_freq_map):
    """Test that build_feature_vector returns correct dimensions and uses defaults."""
    vec = build_feature_vector(
        bedrooms=2,
        bathrooms=1.5,
        accommodates=4,
        room_type="Entire home/apt",
        neighbourhood="Ward 115",
        availability_365=365,
        is_superhost=True,
        host_verified=False,
        num_amenities=None,
        amenities=None,
        neigh_freq_map=dummy_freq_map,
    )
    
    assert len(vec) == len(FEATURE_COLUMNS)
    # Bedrooms
    assert vec[0] == 2.0
    # Bathrooms
    assert vec[1] == 1.5
    # Accommodates
    assert vec[2] == 4
    # One-hot: Entire home/apt activated
    assert vec[3] == 1
    assert vec[4] == 0
    assert vec[5] == 0
    assert vec[6] == 0
    # Neighbourhood freq lookup
    assert vec[7] == 0.05
    # Default amenities count
    assert vec[8] == 10
    # Availability ratio: 365/365
    assert vec[9] == 1.0
    # Boolean conversions
    assert vec[10] == 1
    assert vec[11] == 0

def test_clamping_out_of_bounds_inputs(dummy_freq_map):
    """Test that inputs exceeding valid limits are correctly clamped (M6)."""
    # Test values far outside sensible bounds
    vec = build_feature_vector(
        bedrooms=25,          # Max clamp 10
        bathrooms=15.0,       # Max clamp 10.0
        accommodates=100,     # Max clamp 20
        room_type="Private room",
        neighbourhood="Camps Bay",
        availability_365=1000, # Max clamp 365
        is_superhost=False,
        host_verified=True,
        num_amenities=200,    # Max clamp 100
        amenities=None,
        neigh_freq_map=dummy_freq_map,
    )
    
    assert vec[0] == 10.0
    assert vec[1] == 10.0
    assert vec[2] == 20
    # One-hot: Private room activated
    assert vec[3] == 0
    assert vec[4] == 1
    assert vec[5] == 0
    assert vec[6] == 0
    assert vec[7] == 0.02
    assert vec[8] == 100
    assert vec[9] == 1.0 # Clamped 1000/365 to 365/365

def test_amenities_reconciliation(dummy_freq_map):
    """Test various input types for num_amenities and list-based amenities (C3)."""
    # 1. Passed as direct integer
    vec_int = build_feature_vector(
        bedrooms=1, bathrooms=1.0, accommodates=2, room_type="Private room",
        neighbourhood="Ward 115", availability_365=100, is_superhost=False, host_verified=False,
        num_amenities=15, amenities=None, neigh_freq_map=dummy_freq_map
    )
    assert vec_int[8] == 15

    # 2. Passed as a list of strings
    vec_list = build_feature_vector(
        bedrooms=1, bathrooms=1.0, accommodates=2, room_type="Private room",
        neighbourhood="Ward 115", availability_365=100, is_superhost=False, host_verified=False,
        num_amenities=None, amenities=["Wifi", "Pool", "Kitchen", "Air Conditioning"],
        neigh_freq_map=dummy_freq_map
    )
    assert vec_list[8] == 4

    # 3. Passed as a comma-separated string
    vec_str = build_feature_vector(
        bedrooms=1, bathrooms=1.0, accommodates=2, room_type="Private room",
        neighbourhood="Ward 115", availability_365=100, is_superhost=False, host_verified=False,
        num_amenities=None, amenities="Wifi,Pool,Kitchen", neigh_freq_map=dummy_freq_map
    )
    assert vec_str[8] == 3
