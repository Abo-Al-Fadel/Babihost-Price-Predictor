# BabiHost Accommodation Price Predictor

**Live App:** [https://babihost.streamlit.app/](https://babihost.streamlit.app/)

ML project for BabiHost internship. Predicts nightly prices using Cape Town Airbnb data as proxy for West Africa. Includes data cleaning, feature engineering, hyperparameter tuning, Flask API, and Streamlit dashboard.

## Setup
1. Clone repo, create virtual environment.
2. `pip install -r requirements.txt`
3. Download `listings.csv` from [Inside Airbnb Cape Town](http://insideairbnb.com/get-the-data.html) → place in `data/`
4. Run notebooks in order: `01_EDA_and_cleaning.ipynb` → `02_feature_engineering.ipynb` → `03_model_tuning.ipynb`
5. Start API: `cd app && python app.py`
6. Start dashboard (bonus): `streamlit run streamlit/dashboard.py`

## Model Performance
- **Best model:** Tuned Random Forest (R² = 0.22, MAE = $122)
- **Features (9):** bedrooms, bathrooms, accommodates, room_type, neighbourhood (ward), num_amenities, availability, superhost, verified host
- **Limitation:** Dataset contains only premium listings ($180–$999). For budget predictions, retrain with broader data.

## API Example (Python)
```python
import requests
r = requests.post('http://127.0.0.1:5000/predict_price', json={
    'bedrooms':2, 'bathrooms':1.5, 'accommodates':4, 'room_type':'Entire home/apt',
    'neighbourhood':'Ward 115', 'num_amenities':20, 'availability_365':300,
    'is_superhost':True, 'host_verified':True
})
print(r.json())
```

## Repository Structure
```text
BabiHost-Price-Predictor/
├── app/                # Flask API + saved model artifacts (.pkl)
├── data/               # listings.csv (download manually)
├── notebooks/          # 3 Jupyter notebooks (EDA → Features → Tuning)
├── src/                # Reusable Python modules
│   ├── features.py     # Feature transformation & model loading
│   ├── preprocess.py   # Data cleaning utilities
│   └── train_model.py  # Model training & evaluation helpers
├── streamlit/          # Interactive dashboard
├── requirements.txt    # Production dependencies
└── README.md           # This file
```