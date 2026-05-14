import os
import sys
from flask import Flask, request, jsonify

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features import (
    load_model,
    load_neighbourhood_freq,
    build_feature_vector,
    predict_price,
)

app = Flask(__name__)

model = load_model()
neigh_freq = load_neighbourhood_freq()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'BabiHost price predictor is alive'})

@app.route('/predict_price', methods=['POST'])
def predict():
    data = request.get_json()

    features = build_feature_vector(
        bedrooms=data.get('bedrooms', 1),
        bathrooms=data.get('bathrooms', 1.0),
        accommodates=data.get('accommodates', 2),
        room_type=data.get('room_type', 'Private room'),
        neighbourhood=data.get('neighbourhood', 'Ward 115'),
        num_amenities=data.get('num_amenities', 10),
        availability_365=data.get('availability_365', 200),
        is_superhost=data.get('is_superhost', False),
        host_verified=data.get('host_verified', False),
        neigh_freq_map=neigh_freq,
    )

    predicted = predict_price(model, features)
    return jsonify({
        'predicted_nightly_price_usd': predicted,
        'currency': 'USD'
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)