# BabiHost Accommodation Price Predictor

**Live App:** [https://babihost.streamlit.app/](https://babihost.streamlit.app/)

Machine learning project for the BabiHost internship. This repository implements an end-to-end price prediction system using Cape Town Airbnb listings as a representative proxy for short-term rental properties in West Africa.

The pipeline comprises source code modules, hyperparameter tuning scripts, an automated training runner, a hardened Flask inference API, and an interactive Streamlit dashboard.

---

## Architectural Layout

```text
BabiHost-Price-Predictor/
├── .devcontainer/      # DevContainer environment configs
├── .github/            # GitHub CI/CD workflows
├── app/                # Flask REST API & persisted model pickles
├── data/               # Listings source datasets
├── notebooks/          # Refactored Jupyter research notebooks
├── src/                # Central processing & inference modules
│   ├── constants.py    # Common mappings & feature orders
│   ├── features.py     # Feature vector construction & interval math
│   ├── preprocess.py   # Price outlier cleaning & parsing
│   └── train_model.py  # Sklearn training wrappers
├── streamlit/          # Dashboard visualization code
├── tests/              # Pytest unit & integration test suites
├── Dockerfile          # Production container setup
├── LICENSE             # Open-source MIT License
├── Procfile            # Deployment runner command file
├── README.md           # Documentation
├── rebuild_model.py    # CLI script to retrain the pipeline
└── requirements.txt    # Pinned production packages
```

---

## Model Pipeline & Performance

### Data Selection & Outlier Logic
Previous iterations incorrectly treated Cape Town's South African Rand (ZAR) prices as USD, leading to a selection bias filter (10 to 1000) that dropped 83% of the representative data.
We corrected the filtering boundaries to:
- Minimum listing price: 100 ZAR (approx. 5.50 USD)
- Maximum listing price: 10,000 ZAR (approx. 550 USD)

This correction increased the clean training sample count from 4,480 to 18,453 listings.

### Feature Processing
The model consumes 12 features using a whitelisted one-hot encoding schema:
1. `bedrooms` (clamped [0, 10])
2. `bathrooms` (clamped [0.5, 10.0])
3. `accommodates` (clamped [1, 20])
4. `room_type_entire` (One-hot entire home)
5. `room_type_private` (One-hot private room)
6. `room_type_shared` (One-hot shared room)
7. `room_type_hotel` (One-hot hotel room)
8. `neighbourhood_cleansed` (Frequency encoded ward ratio)
9. `num_amenities` (Extracted counts clamped [0, 100])
10. `availability_ratio` (Clamped ratio of availability_365)
11. `is_superhost` (Boolean flag)
12. `host_identity_verified` (Boolean flag)

### Model Statistics
- **Architecture**: Tuned Random Forest Regressor
- **R-squared Score**: 0.6186 (up from 0.22 in the baseline)
- **Mean Absolute Error (MAE)**: 766.35 ZAR (approx. 42.50 USD)
- **Uncertainty Quantification**: Returns a 95% confidence range calculated from the prediction variance across individual trees in the random forest ensemble.

---

## Deployment & Setup Instructions

### Local Development Setup
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Abo-Al-Fadel/Babihost-Price-Predictor.git
   cd Babihost-Price-Predictor
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the automated model rebuild CLI to recreate cleaned data, calculate frequency maps, and train/tune the Random Forest model:
   ```bash
   python rebuild_model.py
   ```

### Docker Execution
To run the Flask API inside a container:
```bash
# Build image
docker build -t babihost-api .

# Run container exposing port 5000
docker run -p 5000:5000 -e BABIHOST_API_KEY=your_secure_api_key babihost-api
```

---

## Running Quality Checks (Test Suite)

Run the test suite using pytest to verify both features parsing and the API client:
```bash
python -m pytest
```

---

## Hardened REST API Specification

The Flask API runs at port 5000. It requires all request calls to supply an `X-API-Key` header for authorization.

### Endpoint: GET /health
Liveness probe checking application status.
- **Headers**: None
- **Response (200)**:
  ```json
  {
    "status": "healthy",
    "service": "BabiHost price predictor",
    "model_loaded": true,
    "timestamp": 1779262751.46
  }
  ```

### Endpoint: GET /neighbourhoods
Discover all supported ward names for prediction payloads.
- **Headers**: `X-API-Key: <your_secret_key>`
- **Response (200)**:
  ```json
  {
    "status": "success",
    "neighbourhoods": ["Ward 1", "Ward 2", "..."]
  }
  ```

### Endpoint: POST /predict_price
Calculate optimal price with prediction bounds.
- **Headers**: `X-API-Key: <your_secret_key>`, `Content-Type: application/json`
- **Payload**:
  ```json
  {
    "bedrooms": 2.0,
    "bathrooms": 1.5,
    "accommodates": 4,
    "room_type": "Entire home/apt",
    "neighbourhood": "Ward 115",
    "availability_365": 300,
    "is_superhost": true,
    "host_verified": true,
    "amenities": ["Wifi", "Kitchen", "Air Conditioning", "Pool"]
  }
  ```
- **Response (200)**:
  ```json
  {
    "status": "success",
    "predicted_nightly_price": 2450.50,
    "lower_bound": 1820.25,
    "upper_bound": 3080.75,
    "currency": "ZAR",
    "note": "Nightly price in Cape Town proxy currency (ZAR)."
  }
  ```

---

## Streamlit Dashboard UI

Run the interactive dashboard locally:
```bash
streamlit run streamlit/dashboard.py
```
This launches a browser session on `http://127.0.0.1:8501`. Enter property parameters to instantly view nightly predictions alongside their 95% confidence intervals and approximate USD conversions.