import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
from src.features import (
    ROOM_TYPE_MAP,
    load_model,
    load_neighbourhood_freq,
    build_feature_vector,
    predict_price,
)

st.set_page_config(page_title="BabiHost Price Predictor", layout="centered")
st.title("BabiHost Price Estimator \n   Cape Town Proxy")

model = load_model()
neigh_freq = load_neighbourhood_freq()

st.sidebar.header("Property Details")
bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 2)
bathrooms = st.sidebar.number_input("Bathrooms (e.g., 1.5)", 1.0, 10.0, 2.0, 0.5)
accommodates = st.sidebar.number_input("Accommodates", 1, 20, 4)
room_type = st.sidebar.selectbox("Room Type", list(ROOM_TYPE_MAP.keys()))
neighbourhood = st.sidebar.selectbox("Neighbourhood (Ward)", list(neigh_freq.keys()))
num_amenities = st.sidebar.slider("Total Amenities Count", 1, 50, 20)
availability_365 = st.sidebar.slider("Available Days/Year", 0, 365, 300)
is_superhost = st.sidebar.checkbox("Superhost")
host_verified = st.sidebar.checkbox("Host Verified")

if st.sidebar.button("Predict Price"):
    features = build_feature_vector(
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        accommodates=accommodates,
        room_type=room_type,
        neighbourhood=neighbourhood,
        num_amenities=num_amenities,
        availability_365=availability_365,
        is_superhost=is_superhost,
        host_verified=host_verified,
        neigh_freq_map=neigh_freq,
    )
    pred = predict_price(model, features)
    st.success(f"Predicted nightly price: **${pred:.2f} USD**")
    st.caption("Model trained on premium Cape Town listings ($180–$999).")