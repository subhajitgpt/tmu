"""
Lightweight ML Service for n8n Integration
This minimal Flask API serves only the ML model prediction endpoint.
n8n handles all orchestration, business logic, and integrations.
"""
from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd
from catboost import Pool

app = Flask(__name__)

# Model path
MODEL_PATH = r"C:\TMU\credit_risk_model.pkl"
MODEL = None

# Optional simple auth (recommended when exposing via tunnel)
# If ML_API_KEY is set, callers must send header: X-ML-API-Key: <value>
ML_API_KEY = os.getenv("ML_API_KEY")
ML_API_KEY_HEADER = "X-ML-API-Key"

# Features expected by the model
FEATURES = [
    "utilisation", "dpd_days", "cash_credit_ratio", "cash_debit_ratio",
    "inbound_cheque_bounce_count", "inbound_cheque_bounce_amt",
    "outbound_cheque_bounce_count", "outbound_cheque_bounce_amt",
    "total_amt_credit", "total_amt_debit", "no_of_banks"
]


def load_model():
    """Load the trained CatBoost model."""
    global MODEL
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            MODEL = pickle.load(f)
        print(f"✓ Model loaded from {MODEL_PATH}")
        return True
    else:
        print(f"⚠ Error: Model file {MODEL_PATH} not found")
        return False


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "service": "ML Prediction Service for n8n",
        "auth_enabled": bool(ML_API_KEY)
    }), 200


def _require_api_key_if_configured():
    if not ML_API_KEY:
        return None
    provided = request.headers.get(ML_API_KEY_HEADER)
    if not provided or provided != ML_API_KEY:
        return jsonify({
            "error": "Unauthorized",
            "details": f"Missing or invalid {ML_API_KEY_HEADER}"
        }), 401
    return None


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict risk probability for a single customer.
    
    Expected JSON:
    {
        "customer_id": "CUST123",
        "features": {
            "utilisation": 15.5,
            "dpd_days": 45,
            ...
        }
    }
    
    Returns:
    {
        "customer_id": "CUST123",
        "probability": 0.3521,
        "shap_values": [0.012, -0.034, ...]
    }
    """
    if MODEL is None:
        return jsonify({
            "error": "Model not loaded. Check server startup logs."
        }), 503

    auth_error = _require_api_key_if_configured()
    if auth_error is not None:
        return auth_error
    
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                "error": "Missing 'features' in request body"
            }), 400
        
        customer_id = data.get('customer_id', 'N/A')
        feature_values = data['features']
        
        # Ensure all features are present
        for f in FEATURES:
            if f not in feature_values:
                return jsonify({
                    "error": f"Missing feature: {f}"
                }), 400
        
        # Prepare DataFrame
        X = pd.DataFrame([{f: float(feature_values[f]) for f in FEATURES}])
        
        # Get prediction probability
        probability = float(MODEL.predict_proba(X)[:, 1][0])
        
        # Get SHAP values for explainability
        shap_values = MODEL.get_feature_importance(Pool(X), type="ShapValues")
        shap_contrib = shap_values[0, :-1].tolist()  # Exclude bias term
        
        # Build response
        response = {
            "customer_id": customer_id,
            "probability": round(probability, 4),
            "shap_values": shap_contrib
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict risk probability for multiple customers.
    
    Expected JSON:
    {
        "customers": [
            {
                "customer_id": "CUST123",
                "features": {...}
            },
            ...
        ]
    }
    
    Returns:
    {
        "results": [
            {
                "customer_id": "CUST123",
                "probability": 0.3521
            },
            ...
        ]
    }
    """
    if MODEL is None:
        return jsonify({
            "error": "Model not loaded"
        }), 503

    auth_error = _require_api_key_if_configured()
    if auth_error is not None:
        return auth_error
    
    try:
        data = request.get_json()
        
        if not data or 'customers' not in data:
            return jsonify({
                "error": "Missing 'customers' array in request body"
            }), 400
        
        customers = data['customers']
        results = []
        
        for customer in customers:
            try:
                customer_id = customer.get('customer_id', 'N/A')
                feature_values = customer['features']
                
                # Prepare DataFrame
                X = pd.DataFrame([{f: float(feature_values[f]) for f in FEATURES}])
                
                # Get prediction
                probability = float(MODEL.predict_proba(X)[:, 1][0])
                
                results.append({
                    "customer_id": customer_id,
                    "probability": round(probability, 4)
                })
            
            except Exception as e:
                results.append({
                    "customer_id": customer.get('customer_id', 'N/A'),
                    "error": str(e)
                })
        
        return jsonify({"results": results}), 200
    
    except Exception as e:
        return jsonify({
            "error": "Batch prediction failed",
            "details": str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("ML Prediction Service for n8n Integration")
    print("=" * 60)
    
    # Load model
    if load_model():
        print("✓ Service ready to accept predictions")
        print(f"✓ Listening on http://localhost:5064")
        print(f"✓ Endpoints:")
        print(f"  - GET  /health")
        print(f"  - POST /predict")
        print(f"  - POST /predict/batch")
        print("=" * 60)
        
        # Run on different port than main API
        app.run(host="0.0.0.0", port=5064, debug=False)
    else:
        print("✗ Failed to load model. Service not started.")
        print("  Run 'python train_and_save_model.py' first to create the model.")
