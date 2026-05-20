import os
import sys

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
from src.constants import ROOM_TYPES
from src.features import (
    load_model,
    load_neighbourhood_freq,
    build_feature_vector,
    predict_price,
)

# Page Layout & Styling
st.set_page_config(
    page_title="BabiHost Accommodation Price Estimator",
    page_icon="house",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Header
st.title("BabiHost Price Estimator")
st.subheader("Cape Town Proxy for West African Short-Term Rentals")
st.markdown("""
Welcome to the BabiHost price estimator dashboard!
This model leverages Cape Town Airbnb listing statistics as a highly robust proxy to estimate optimal nightly accommodation pricing. 
""")

# Load artifacts
@st.cache_resource
def load_cached_artifacts():
    return load_model(), load_neighbourhood_freq()

try:
    model, neigh_freq = load_cached_artifacts()
    artifact_loaded = True
except Exception as e:
    st.error(f"Error loading model artifacts: {str(e)}")
    artifact_loaded = False

if artifact_loaded:
    # Sidebar Input Fields
    st.sidebar.header("Property Details")
    bedrooms = st.sidebar.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
    bathrooms = st.sidebar.number_input("Bathrooms (e.g., 1.5)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
    accommodates = st.sidebar.number_input("Accommodates (Guests)", min_value=1, max_value=20, value=4, step=1)
    room_type = st.sidebar.selectbox("Room Type", ROOM_TYPES)
    
    # Sort neighbourhoods alphabetically for discoverability
    sorted_wards = sorted(list(neigh_freq.keys()))
    neighbourhood = st.sidebar.selectbox("Neighbourhood (Ward)", sorted_wards)
    
    num_amenities = st.sidebar.slider("Total Amenities Count", min_value=0, max_value=100, value=20)
    availability_365 = st.sidebar.slider("Available Days per Year", min_value=0, max_value=365, value=300)
    is_superhost = st.sidebar.checkbox("Host is a Superhost")
    host_verified = st.sidebar.checkbox("Host Identity is Verified")

    st.sidebar.markdown("---")
    st.sidebar.caption("Model Version: **Random Forest v2.0 (One-Hot Encoded, R²≈0.62)**")

    # Predict Trigger
    if st.sidebar.button("Predict Optimal Price", use_container_width=True):
        features = build_feature_vector(
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            accommodates=accommodates,
            room_type=room_type,
            neighbourhood=neighbourhood,
            availability_365=availability_365,
            is_superhost=is_superhost,
            host_verified=host_verified,
            num_amenities=num_amenities,
            neigh_freq_map=neigh_freq,
        )

        # Run prediction
        res = predict_price(model, features)
        pred = res['predicted_price']
        low = res['lower_bound']
        high = res['upper_bound']

        # Conversions
        usd_exch = 18.0
        pred_usd = pred / usd_exch
        low_usd = low / usd_exch
        high_usd = high / usd_exch

        st.markdown("### Price Prediction Results")

        # Display clean metrics in columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted Nightly Price (ZAR)", value=f"{pred:,.2f} ZAR")
        with col2:
            st.metric(label="Estimated Price (USD)", value=f"${pred_usd:,.2f} USD")

        # Clean display of ranges without nested bold tags that confuse markdown parsers
        st.info(f"**95% Confidence Range (ZAR):** {low:,.2f} ZAR to {high:,.2f} ZAR")
        st.success(f"**95% Confidence Range (USD):** ${low_usd:,.2f} USD to ${high_usd:,.2f} USD")

        st.caption(f"Note: Calculations use an approximate exchange rate of 1 USD = {usd_exch} ZAR. The estimator captures normal mid/premium short-term rentals within range (100 to 10,000 ZAR).")
else:
    st.warning("Prediction dashboard is disabled because model pickle files were not found. Please train the model to enable predicting.")