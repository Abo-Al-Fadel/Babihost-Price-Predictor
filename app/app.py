import os
import sys
import time
import logging
from functools import wraps
from collections import defaultdict
from threading import Lock
from typing import Optional, List, Union
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, field_validator, ValidationError

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.constants import ROOM_TYPES, FEATURE_COLUMNS
from src.features import (
    load_model,
    load_neighbourhood_freq,
    build_feature_vector,
    predict_price,
)

# ── Configure Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BabiHostAPI")

app = Flask(__name__)

# ── Global State Setup ────────────────────────────────────────────────────────
try:
    model = load_model()
    neigh_freq = load_neighbourhood_freq()
    logger.info("Successfully loaded tuned model and neighbourhood frequency mapping.")
except Exception as e:
    logger.critical(f"Failed to load model artifacts on startup: {str(e)}")
    # We do not crash startup here to allow testing/health routes to load, but predictions will fail gracefully.
    model = None
    neigh_freq = {}

# ── Security & Authentication ──────────────────────────────────────────────────
API_KEY = os.environ.get("BABIHOST_API_KEY", "babihost_demo_secret_key")

def require_api_key(f):
    """Decorator to require a valid X-API-Key header."""
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if not key or key != API_KEY:
            logger.warning(f"Unauthorized access attempt from IP: {request.remote_addr} - Key provided: {bool(key)}")
            return jsonify({
                "status": "error",
                "message": "Unauthorized: Invalid or missing X-API-Key header."
            }), 401
        return f(*args, **kwargs)
    return decorated

# ── Thread-Safe Rate Limiter ───────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, limit: int = 60, window: int = 60):
        self.limit = limit
        self.window = window
        self.requests = defaultdict(list)
        self.lock = Lock()

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        with self.lock:
            # Clear requests older than the sliding window
            self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < self.window]
            if len(self.requests[client_ip]) >= self.limit:
                return False
            self.requests[client_ip].append(now)
            return True

# Initialize rate limiter: 60 requests per minute
limiter = RateLimiter(limit=60, window=60)

def rate_limit(f):
    """Decorator to apply client IP-based rate limiting."""
    @wraps(f)
    def decorated(*args, **kwargs):
        client_ip = request.remote_addr or "unknown_ip"
        if not limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for client IP: {client_ip}")
            return jsonify({
                "status": "error",
                "message": "Too Many Requests: Rate limit of 60 requests per minute exceeded."
            }), 429
        return f(*args, **kwargs)
    return decorated

# ── Pydantic Request Schema ────────────────────────────────────────────────────
class PricePredictionRequest(BaseModel):
    bedrooms: float = Field(..., ge=0.0, le=10.0, description="Number of bedrooms (0 to 10)")
    bathrooms: float = Field(..., ge=0.5, le=10.0, description="Number of bathrooms (0.5 to 10)")
    accommodates: int = Field(..., ge=1, le=20, description="Number of guests accommodated (1 to 20)")
    room_type: str = Field(..., description="One of the supported room types")
    neighbourhood: str = Field(..., description="Supported Cape Town ward/neighbourhood cleansed")
    availability_365: float = Field(..., ge=0.0, le=365.0, description="Available days in a year (0 to 365)")
    is_superhost: bool = Field(..., description="Whether host is a superhost")
    host_verified: bool = Field(..., description="Whether host identity is verified")
    num_amenities: Optional[int] = Field(None, ge=0, le=100, description="Precalculated total number of amenities")
    amenities: Optional[Union[List[str], str]] = Field(None, description="List or comma-separated string of amenities to compute counts server-side")

    @field_validator('room_type')
    @classmethod
    def validate_room_type(cls, v: str) -> str:
        if v not in ROOM_TYPES:
            raise ValueError(f"room_type must be one of {ROOM_TYPES}")
        return v

    @field_validator('neighbourhood')
    @classmethod
    def validate_neighbourhood(cls, v: str) -> str:
        if neigh_freq and v not in neigh_freq:
            raise ValueError(f"neighbourhood '{v}' is unknown. Please choose a valid neighbourhood cleansing ward.")
        return v

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Liveness probe to monitor service status."""
    status_info = {
        'status': 'healthy',
        'service': 'BabiHost price predictor',
        'model_loaded': model is not None,
        'timestamp': time.time()
    }
    return jsonify(status_info), 200


@app.route('/neighbourhoods', methods=['GET'])
@require_api_key
@rate_limit
def get_neighbourhoods():
    """Discover all supported neighbourhood names (wards) to avoid magic values (m7)."""
    return jsonify({
        "status": "success",
        "neighbourhoods": sorted(list(neigh_freq.keys())) if neigh_freq else []
    }), 200


@app.route('/predict_price', methods=['POST'])
@require_api_key
@rate_limit
def predict():
    """
    Predict nightly price of short-term accommodation.
    Requires header X-API-Key and a valid JSON body matching PricePredictionRequest.
    """
    if not model:
        logger.error("Prediction failed: Tuned model is not loaded.")
        return jsonify({
            "status": "error",
            "message": "Service Unavailable: Model artifacts failed to load. Please check system logs."
        }), 500

    # Ensure JSON payload is present
    data = request.get_json(silent=True)
    if data is None:
        logger.warning("Prediction request rejected: payload is not valid JSON or Content-Type is missing.")
        return jsonify({
            "status": "error",
            "message": "Bad Request: Missing or malformed JSON payload. Content-Type must be application/json."
        }), 400

    # Validate inputs using Pydantic
    try:
        validated_data = PricePredictionRequest(**data)
    except ValidationError as err:
        # Extract only JSON-serializable details to avoid TypeError on ValueError contexts
        clean_errors = []
        for error in err.errors():
            clean_errors.append({
                "loc": error.get("loc"),
                "msg": error.get("msg"),
                "type": error.get("type")
            })
        logger.warning(f"Validation error on prediction input: {clean_errors}")
        return jsonify({
            "status": "error",
            "message": "Validation Error: Input parameters failed schema validation.",
            "details": clean_errors
        }), 400
    except Exception as e:
        logger.error(f"Unexpected parsing error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Bad Request: Failed to parse input payload. {str(e)}"
        }), 400

    # Run inference
    try:
        # Build feature vector incorporating clamping and amenities counting
        features = build_feature_vector(
            bedrooms=validated_data.bedrooms,
            bathrooms=validated_data.bathrooms,
            accommodates=validated_data.accommodates,
            room_type=validated_data.room_type,
            neighbourhood=validated_data.neighbourhood,
            availability_365=validated_data.availability_365,
            is_superhost=validated_data.is_superhost,
            host_verified=validated_data.host_verified,
            num_amenities=validated_data.num_amenities,
            amenities=validated_data.amenities,
            neigh_freq_map=neigh_freq,
        )

        prediction_result = predict_price(model, features)
        
        logger.info(
            f"Prediction success for IP {request.remote_addr} - "
            f"Rooms: {validated_data.bedrooms}, Guests: {validated_data.accommodates}, "
            f"Ward: {validated_data.neighbourhood} -> "
            f"Estimate: {prediction_result['predicted_price']} ZAR "
            f"[{prediction_result['lower_bound']} - {prediction_result['upper_bound']}]"
        )

        return jsonify({
            "status": "success",
            "predicted_nightly_price": prediction_result['predicted_price'],
            "lower_bound": prediction_result['lower_bound'],
            "upper_bound": prediction_result['upper_bound'],
            "currency": "ZAR",
            "note": "Nightly price in Cape Town proxy currency (ZAR)."
        }), 200

    except Exception as exc:
        logger.exception(f"Internal prediction execution crash: {str(exc)}")
        return jsonify({
            "status": "error",
            "message": "Internal Server Error: An unexpected error occurred while executing inference."
        }), 500


if __name__ == '__main__':
    # Served locally on Werkzeug for testing. Gunicorn should be used in production.
    app.run(debug=False, host='0.0.0.0', port=5000)