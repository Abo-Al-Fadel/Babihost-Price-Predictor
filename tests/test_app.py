import os
import sys
import json
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set dummy API key for testing
os.environ["BABIHOST_API_KEY"] = "test_secret_key"

from app.app import app, API_KEY

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test the public liveness health route."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'model_loaded' in data

def test_unauthorized_predict(client):
    """Test that missing X-API-Key header returns 401 (C4)."""
    response = client.post('/predict_price', json={})
    assert response.status_code == 401
    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert 'Unauthorized' in data['message']

def test_get_neighbourhoods(client):
    """Test the discoverability /neighbourhoods endpoint (m7)."""
    response = client.get('/neighbourhoods', headers={"X-API-Key": "test_secret_key"})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert isinstance(data['neighbourhoods'], list)
    assert len(data['neighbourhoods']) > 0

def test_validation_errors(client):
    """Test that missing inputs, empty dicts, or invalid data types yield 400 errors (C1, C2)."""
    headers = {"X-API-Key": "test_secret_key"}

    # 1. Test empty body (missing required fields)
    response = client.post('/predict_price', json={}, headers=headers)
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert 'Validation Error' in data['message']

    # 2. Test invalid data types
    payload_bad_types = {
        "bedrooms": "three", # string instead of float
        "bathrooms": 2.0,
        "accommodates": 4,
        "room_type": "Private room",
        "neighbourhood": "Ward 115",
        "availability_365": 200
    }
    response = client.post('/predict_price', json=payload_bad_types, headers=headers)
    assert response.status_code == 400
    
    # 3. Test invalid categoricals / whitelisting
    payload_bad_cat = {
        "bedrooms": 2.0,
        "bathrooms": 2.0,
        "accommodates": 4,
        "room_type": "Magical Castle", # Bad room type
        "neighbourhood": "Ward 115",
        "availability_365": 200
    }
    response = client.post('/predict_price', json=payload_bad_cat, headers=headers)
    assert response.status_code == 400
    
    # 4. Test bad neighbourhood ward
    payload_bad_ward = {
        "bedrooms": 2.0,
        "bathrooms": 2.0,
        "accommodates": 4,
        "room_type": "Private room",
        "neighbourhood": "Not A Real Ward", # Bad ward
        "availability_365": 200
    }
    response = client.post('/predict_price', json=payload_bad_ward, headers=headers)
    assert response.status_code == 400

def test_successful_prediction(client):
    """Test successful price prediction with intervals (M1)."""
    headers = {"X-API-Key": "test_secret_key"}
    payload = {
        "bedrooms": 2,
        "bathrooms": 1.5,
        "accommodates": 4,
        "room_type": "Entire home/apt",
        "neighbourhood": "Ward 115",
        "availability_365": 250,
        "is_superhost": True,
        "host_verified": True,
        "amenities": ["Wifi", "Kitchen", "Air Conditioning", "Pool"]
    }
    
    response = client.post('/predict_price', json=payload, headers=headers)
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert data['status'] == 'success'
    assert 'predicted_nightly_price' in data
    assert 'lower_bound' in data
    assert 'upper_bound' in data
    assert data['currency'] == 'ZAR'
    
    # Check that bounds make mathematical sense
    assert data['lower_bound'] <= data['predicted_nightly_price']
    assert data['predicted_nightly_price'] <= data['upper_bound']

def test_rate_limiting(client):
    """Test that sending requests in high volume triggers a 429 Too Many Requests (C4)."""
    headers = {"X-API-Key": "test_secret_key"}
    
    # Send 70 requests rapidly; limit is 60.
    triggered_429 = False
    for _ in range(70):
        response = client.get('/neighbourhoods', headers=headers)
        if response.status_code == 429:
            triggered_429 = True
            data = json.loads(response.data)
            assert data['status'] == 'error'
            assert 'Rate limit' in data['message']
            break
            
    assert triggered_429
