"""
Credit Scoring API
This Flask API provides real-time credit risk scoring for customers
using a pre-trained CatBoost model saved as a .pkl file.
"""
from __future__ import annotations
import pickle
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from catboost import Pool

app = Flask(__name__)

# Model path
MODEL_PATH = "credit_risk_model.pkl"

# Features expected by the model
FEATURES = [
    "utilisation", "dpd_days", "cash_credit_ratio", "cash_debit_ratio",
    "inbound_cheque_bounce_count", "inbound_cheque_bounce_amt",
    "outbound_cheque_bounce_count", "outbound_cheque_bounce_amt",
    "total_amt_credit", "total_amt_debit", "no_of_banks"
]

# Load model at startup
MODEL = None


def load_model():
    """Load the trained model from pickle file."""
    global MODEL
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            MODEL = pickle.load(f)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠ Warning: Model file {MODEL_PATH} not found. Train and save model first.")


def get_risk_bucket(probability: float) -> str:
    """
    Categorize risk based on probability score.
    
    Args:
        probability: Risk probability (0-1)
    
    Returns:
        Risk bucket category
    """
    if probability >= 0.90:
        return "Very High"
    elif probability >= 0.80:
        return "High"
    elif probability >= 0.70:
        return "Moderate"
    elif probability >= 0.50:
        return "Low"
    else:
        return "No Risk"


def get_decision(probability: float, bucket: str) -> Dict[str, Any]:
    """
    Make credit decision based on risk probability and bucket.
    
    Args:
        probability: Risk probability (0-1)
        bucket: Risk bucket category
    
    Returns:
        Decision dictionary with status, recommendation, and reasoning
    """
    if probability >= 0.80:
        return {
            "status": "REJECT",
            "recommendation": "Application rejected due to high credit risk",
            "risk_level": "Critical",
            "action": "Deny credit immediately",
            "explanation": (
                f"Customer shows a {probability*100:.1f}% probability of default, "
                f"classified as '{bucket}' risk. This exceeds acceptable risk thresholds. "
                "Key risk indicators suggest high likelihood of payment default."
            )
        }
    elif probability >= 0.70:
        return {
            "status": "REVIEW",
            "recommendation": "Requires manual review by credit officer",
            "risk_level": "Elevated",
            "action": "Conduct detailed assessment with additional documentation",
            "explanation": (
                f"Customer shows a {probability*100:.1f}% probability of default, "
                f"classified as '{bucket}' risk. This warrants careful evaluation. "
                "Consider requesting additional collateral or guarantors."
            )
        }
    elif probability >= 0.50:
        return {
            "status": "CONDITIONAL_ACCEPT",
            "recommendation": "Accept with enhanced monitoring",
            "risk_level": "Moderate",
            "action": "Approve with stricter terms (lower limit, higher interest rate)",
            "explanation": (
                f"Customer shows a {probability*100:.1f}% probability of default, "
                f"classified as '{bucket}' risk. Acceptable with risk mitigation measures. "
                "Recommend regular monitoring and conservative credit limits."
            )
        }
    else:
        return {
            "status": "ACCEPT",
            "recommendation": "Application approved",
            "risk_level": "Low",
            "action": "Grant credit with standard terms",
            "explanation": (
                f"Customer shows a {probability*100:.1f}% probability of default, "
                f"classified as '{bucket}' risk. This is within acceptable parameters. "
                "Customer profile indicates good creditworthiness."
            )
        }


def get_top_risk_drivers(feature_values: Dict[str, float], shap_values: np.ndarray) -> list:
    """
    Identify top 3 features driving the risk score.
    
    Args:
        feature_values: Dictionary of feature names and values
        shap_values: SHAP values from the model
    
    Returns:
        List of top risk drivers with explanations
    """
    # Get absolute SHAP values and sort
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(-abs_shap)[:3]
    
    drivers = []
    for idx in top_indices:
        feature_name = FEATURES[idx]
        feature_value = feature_values.get(feature_name, 0)
        shap_impact = shap_values[idx]
        
        # Determine impact direction
        impact_direction = "increases" if shap_impact > 0 else "decreases"
        
        drivers.append({
            "feature": feature_name,
            "value": float(feature_value),
            "impact": float(shap_impact),
            "description": f"{feature_name} ({feature_value:.2f}) {impact_direction} risk by {abs(shap_impact):.4f}"
        })
    
    return drivers


def validate_input(data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate input data for required features and value ranges.
    
    Args:
        data: Input data dictionary
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for missing features
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return False, f"Missing required features: {', '.join(missing)}"
    
    # Validate numeric values
    for feature in FEATURES:
        try:
            value = float(data[feature])
            if np.isnan(value) or np.isinf(value):
                return False, f"Feature '{feature}' has invalid value: {data[feature]}"
        except (ValueError, TypeError):
            return False, f"Feature '{feature}' must be numeric, got: {data[feature]}"
    
    # Basic range validations
    if data["utilisation"] < 0 or data["utilisation"] > 100:
        return False, "utilisation must be between 0 and 100"
    
    if data["dpd_days"] < 0:
        return False, "dpd_days cannot be negative"
    
    if data["no_of_banks"] < 0:
        return False, "no_of_banks cannot be negative"
    
    return True, ""


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH
    })


@app.route("/screen", methods=["POST"])
def screen():
    """
    Screen a single customer application.
    
    Expected JSON payload with all required features.
    Returns risk probability, bucket, decision, and explanations.
    """
    if MODEL is None:
        return jsonify({
            "error": "Model not loaded. Please train and save model first."
        }), 503
    
    # Get JSON data
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    # Validate input
    is_valid, error_msg = validate_input(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 400
    
    try:
        # Extract customer ID if provided
        customer_id = data.get("customer_no", "N/A")
        
        # Prepare features
        feature_values = {f: float(data[f]) for f in FEATURES}
        X = pd.DataFrame([feature_values])
        
        # Make prediction
        probability = float(MODEL.predict_proba(X)[:, 1][0])
        bucket = get_risk_bucket(probability)
        
        # Get SHAP values for feature importance
        shap_values = MODEL.get_feature_importance(Pool(X), type="ShapValues")
        shap_contrib = shap_values[0, :-1]  # Exclude bias term
        
        # Get top risk drivers
        risk_drivers = get_top_risk_drivers(feature_values, shap_contrib)
        
        # Get decision
        decision = get_decision(probability, bucket)
        
        # Build response
        response = {
            "customer_id": customer_id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "risk_score": {
                "probability": round(probability, 4),
                "probability_percentage": f"{probability * 100:.2f}%",
                "bucket": bucket
            },
            "decision": decision,
            "risk_drivers": risk_drivers,
            "input_features": feature_values,
            "model_version": "1.0"
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "error": "Internal server error during scoring",
            "details": str(e)
        }), 500


@app.route("/score/batch", methods=["POST"])
def score_batch():
    """
    Score multiple customer applications in a single request.
    
    Expected JSON payload with list of customer data.
    Returns list of scored results.
    """
    if MODEL is None:
        return jsonify({
            "error": "Model not loaded. Please train and save model first."
        }), 503
    
    # Get JSON data
    data = request.get_json()
    if not data or "customers" not in data:
        return jsonify({
            "error": "No JSON data provided or 'customers' key missing"
        }), 400
    
    customers = data["customers"]
    if not isinstance(customers, list):
        return jsonify({"error": "'customers' must be a list"}), 400
    
    results = []
    errors = []
    
    for idx, customer_data in enumerate(customers):
        # Validate input
        is_valid, error_msg = validate_input(customer_data)
        if not is_valid:
            errors.append({
                "index": idx,
                "customer_id": customer_data.get("customer_no", "N/A"),
                "error": error_msg
            })
            continue
        
        try:
            # Extract customer ID
            customer_id = customer_data.get("customer_no", f"Customer_{idx}")
            
            # Prepare features
            feature_values = {f: float(customer_data[f]) for f in FEATURES}
            X = pd.DataFrame([feature_values])
            
            # Make prediction
            probability = float(MODEL.predict_proba(X)[:, 1][0])
            bucket = get_risk_bucket(probability)
            
            # Get decision
            decision = get_decision(probability, bucket)
            
            # Build result
            result = {
                "customer_id": customer_id,
                "risk_score": {
                    "probability": round(probability, 4),
                    "probability_percentage": f"{probability * 100:.2f}%",
                    "bucket": bucket
                },
                "decision": decision
            }
            
            results.append(result)
        
        except Exception as e:
            errors.append({
                "index": idx,
                "customer_id": customer_data.get("customer_no", "N/A"),
                "error": str(e)
            })
    
    response = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_requests": len(customers),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None
    }
    
    return jsonify(response), 200


@app.route("/model/info", methods=["GET"])
def model_info():
    """Get information about the loaded model."""
    if MODEL is None:
        return jsonify({
            "error": "Model not loaded"
        }), 503
    
    return jsonify({
        "model_type": "CatBoostClassifier",
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "model_path": MODEL_PATH,
        "risk_buckets": {
            "Very High": "probability >= 0.90",
            "High": "0.80 <= probability < 0.90",
            "Moderate": "0.70 <= probability < 0.80",
            "Low": "0.50 <= probability < 0.70",
            "No Risk": "probability < 0.50"
        },
        "decision_thresholds": {
            "REJECT": "probability >= 0.80",
            "REVIEW": "0.70 <= probability < 0.80",
            "CONDITIONAL_ACCEPT": "0.50 <= probability < 0.70",
            "ACCEPT": "probability < 0.50"
        }
    })


if __name__ == "__main__":
    # Load model on startup
    load_model()
    
    # Run Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
