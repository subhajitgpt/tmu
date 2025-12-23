# Credit Scoring API - Usage Guide

## Overview
This API provides real-time credit risk assessment for customer applications using a machine learning model.

## Setup

### 1. Train and Save the Model
```bash
python train_and_save_model.py
```
This will:
- Generate training data
- Train a CatBoost classifier
- Save the model as `credit_risk_model.pkl`

### 2. Start the API Server
```bash
python credit_scoring_api.py
```
The API will run on `http://localhost:5000`

## API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "credit_risk_model.pkl"
}
```

---

### Score Single Customer
```http
POST /score
Content-Type: application/json
```

**Request Body:**
```json
{
  "customer_no": "CUST123456",
  "utilisation": 15.5,
  "dpd_days": 45,
  "cash_credit_ratio": 0.18,
  "cash_debit_ratio": 0.15,
  "inbound_cheque_bounce_count": 2,
  "inbound_cheque_bounce_amt": 8500,
  "outbound_cheque_bounce_count": 1,
  "outbound_cheque_bounce_amt": 3200,
  "total_amt_credit": 750000,
  "total_amt_debit": 680000,
  "no_of_banks": 4
}
```

**Response (ACCEPT - Low Risk):**
```json
{
  "customer_id": "CUST123456",
  "timestamp": "2025-12-23T10:30:45.123456",
  "risk_score": {
    "probability": 0.3521,
    "probability_percentage": "35.21%",
    "bucket": "No Risk"
  },
  "decision": {
    "status": "ACCEPT",
    "recommendation": "Application approved",
    "risk_level": "Low",
    "action": "Grant credit with standard terms",
    "explanation": "Customer shows a 35.2% probability of default, classified as 'No Risk' risk. This is within acceptable parameters. Customer profile indicates good creditworthiness."
  },
  "risk_drivers": [
    {
      "feature": "dpd_days",
      "value": 45.0,
      "impact": 0.0234,
      "description": "dpd_days (45.00) increases risk by 0.0234"
    },
    {
      "feature": "utilisation",
      "value": 15.5,
      "impact": 0.0156,
      "description": "utilisation (15.50) increases risk by 0.0156"
    },
    {
      "feature": "inbound_cheque_bounce_amt",
      "value": 8500.0,
      "impact": 0.0089,
      "description": "inbound_cheque_bounce_amt (8500.00) increases risk by 0.0089"
    }
  ],
  "input_features": {
    "utilisation": 15.5,
    "dpd_days": 45,
    "cash_credit_ratio": 0.18,
    "cash_debit_ratio": 0.15,
    "inbound_cheque_bounce_count": 2,
    "inbound_cheque_bounce_amt": 8500,
    "outbound_cheque_bounce_count": 1,
    "outbound_cheque_bounce_amt": 3200,
    "total_amt_credit": 750000,
    "total_amt_debit": 680000,
    "no_of_banks": 4
  },
  "model_version": "1.0"
}
```

**Response (CONDITIONAL_ACCEPT - Moderate Risk):**
```json
{
  "customer_id": "CUST789012",
  "timestamp": "2025-12-23T10:35:22.987654",
  "risk_score": {
    "probability": 0.6234,
    "probability_percentage": "62.34%",
    "bucket": "Low"
  },
  "decision": {
    "status": "CONDITIONAL_ACCEPT",
    "recommendation": "Accept with enhanced monitoring",
    "risk_level": "Moderate",
    "action": "Approve with stricter terms (lower limit, higher interest rate)",
    "explanation": "Customer shows a 62.3% probability of default, classified as 'Low' risk. Acceptable with risk mitigation measures. Recommend regular monitoring and conservative credit limits."
  },
  "risk_drivers": [
    {
      "feature": "dpd_days",
      "value": 120.0,
      "impact": 0.0876,
      "description": "dpd_days (120.00) increases risk by 0.0876"
    },
    {
      "feature": "outbound_cheque_bounce_count",
      "value": 5,
      "impact": 0.0543,
      "description": "outbound_cheque_bounce_count (5.00) increases risk by 0.0543"
    },
    {
      "feature": "utilisation",
      "value": 18.2,
      "impact": 0.0421,
      "description": "utilisation (18.20) increases risk by 0.0421"
    }
  ],
  "input_features": {
    "utilisation": 18.2,
    "dpd_days": 120,
    "cash_credit_ratio": 0.25,
    "cash_debit_ratio": 0.22,
    "inbound_cheque_bounce_count": 3,
    "inbound_cheque_bounce_amt": 25000,
    "outbound_cheque_bounce_count": 5,
    "outbound_cheque_bounce_amt": 18000,
    "total_amt_credit": 450000,
    "total_amt_debit": 520000,
    "no_of_banks": 6
  },
  "model_version": "1.0"
}
```

**Response (REVIEW - Elevated Risk):**
```json
{
  "customer_id": "CUST345678",
  "timestamp": "2025-12-23T10:40:15.456789",
  "risk_score": {
    "probability": 0.7521,
    "probability_percentage": "75.21%",
    "bucket": "Moderate"
  },
  "decision": {
    "status": "REVIEW",
    "recommendation": "Requires manual review by credit officer",
    "risk_level": "Elevated",
    "action": "Conduct detailed assessment with additional documentation",
    "explanation": "Customer shows a 75.2% probability of default, classified as 'Moderate' risk. This warrants careful evaluation. Consider requesting additional collateral or guarantors."
  },
  "risk_drivers": [
    {
      "feature": "dpd_days",
      "value": 180.0,
      "impact": 0.1234,
      "description": "dpd_days (180.00) increases risk by 0.1234"
    },
    {
      "feature": "inbound_cheque_bounce_amt",
      "value": 85000.0,
      "impact": 0.0987,
      "description": "inbound_cheque_bounce_amt (85000.00) increases risk by 0.0987"
    },
    {
      "feature": "cash_credit_ratio",
      "value": 0.32,
      "impact": 0.0765,
      "description": "cash_credit_ratio (0.32) increases risk by 0.0765"
    }
  ],
  "input_features": {
    "utilisation": 19.5,
    "dpd_days": 180,
    "cash_credit_ratio": 0.32,
    "cash_debit_ratio": 0.28,
    "inbound_cheque_bounce_count": 8,
    "inbound_cheque_bounce_amt": 85000,
    "outbound_cheque_bounce_count": 6,
    "outbound_cheque_bounce_amt": 42000,
    "total_amt_credit": 350000,
    "total_amt_debit": 580000,
    "no_of_banks": 8
  },
  "model_version": "1.0"
}
```

**Response (REJECT - High Risk):**
```json
{
  "customer_id": "CUST901234",
  "timestamp": "2025-12-23T10:45:30.123456",
  "risk_score": {
    "probability": 0.8965,
    "probability_percentage": "89.65%",
    "bucket": "High"
  },
  "decision": {
    "status": "REJECT",
    "recommendation": "Application rejected due to high credit risk",
    "risk_level": "Critical",
    "action": "Deny credit immediately",
    "explanation": "Customer shows a 89.6% probability of default, classified as 'High' risk. This exceeds acceptable risk thresholds. Key risk indicators suggest high likelihood of payment default."
  },
  "risk_drivers": [
    {
      "feature": "dpd_days",
      "value": 300.0,
      "impact": 0.2145,
      "description": "dpd_days (300.00) increases risk by 0.2145"
    },
    {
      "feature": "outbound_cheque_bounce_amt",
      "value": 250000.0,
      "impact": 0.1876,
      "description": "outbound_cheque_bounce_amt (250000.00) increases risk by 0.1876"
    },
    {
      "feature": "inbound_cheque_bounce_count",
      "value": 15,
      "impact": 0.1543,
      "description": "inbound_cheque_bounce_count (15.00) increases risk by 0.1543"
    }
  ],
  "input_features": {
    "utilisation": 19.8,
    "dpd_days": 300,
    "cash_credit_ratio": 0.34,
    "cash_debit_ratio": 0.33,
    "inbound_cheque_bounce_count": 15,
    "inbound_cheque_bounce_amt": 180000,
    "outbound_cheque_bounce_count": 12,
    "outbound_cheque_bounce_amt": 250000,
    "total_amt_credit": 200000,
    "total_amt_debit": 850000,
    "no_of_banks": 12
  },
  "model_version": "1.0"
}
```

---

### Score Multiple Customers (Batch)
```http
POST /score/batch
Content-Type: application/json
```

**Request Body:**
```json
{
  "customers": [
    {
      "customer_no": "CUST001",
      "utilisation": 12.3,
      "dpd_days": 15,
      "cash_credit_ratio": 0.10,
      "cash_debit_ratio": 0.08,
      "inbound_cheque_bounce_count": 0,
      "inbound_cheque_bounce_amt": 0,
      "outbound_cheque_bounce_count": 0,
      "outbound_cheque_bounce_amt": 0,
      "total_amt_credit": 900000,
      "total_amt_debit": 800000,
      "no_of_banks": 2
    },
    {
      "customer_no": "CUST002",
      "utilisation": 18.5,
      "dpd_days": 150,
      "cash_credit_ratio": 0.28,
      "cash_debit_ratio": 0.25,
      "inbound_cheque_bounce_count": 7,
      "inbound_cheque_bounce_amt": 65000,
      "outbound_cheque_bounce_count": 4,
      "outbound_cheque_bounce_amt": 35000,
      "total_amt_credit": 400000,
      "total_amt_debit": 600000,
      "no_of_banks": 7
    }
  ]
}
```

**Response:**
```json
{
  "timestamp": "2025-12-23T11:00:00.000000",
  "total_requests": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "customer_id": "CUST001",
      "risk_score": {
        "probability": 0.2345,
        "probability_percentage": "23.45%",
        "bucket": "No Risk"
      },
      "decision": {
        "status": "ACCEPT",
        "recommendation": "Application approved",
        "risk_level": "Low",
        "action": "Grant credit with standard terms",
        "explanation": "Customer shows a 23.5% probability of default, classified as 'No Risk' risk. This is within acceptable parameters. Customer profile indicates good creditworthiness."
      }
    },
    {
      "customer_id": "CUST002",
      "risk_score": {
        "probability": 0.7823,
        "probability_percentage": "78.23%",
        "bucket": "Moderate"
      },
      "decision": {
        "status": "REVIEW",
        "recommendation": "Requires manual review by credit officer",
        "risk_level": "Elevated",
        "action": "Conduct detailed assessment with additional documentation",
        "explanation": "Customer shows a 78.2% probability of default, classified as 'Moderate' risk. This warrants careful evaluation. Consider requesting additional collateral or guarantors."
      }
    }
  ],
  "errors": null
}
```

---

### Get Model Information
```http
GET /model/info
```

**Response:**
```json
{
  "model_type": "CatBoostClassifier",
  "features": [
    "utilisation",
    "dpd_days",
    "cash_credit_ratio",
    "cash_debit_ratio",
    "inbound_cheque_bounce_count",
    "inbound_cheque_bounce_amt",
    "outbound_cheque_bounce_count",
    "outbound_cheque_bounce_amt",
    "total_amt_credit",
    "total_amt_debit",
    "no_of_banks"
  ],
  "feature_count": 11,
  "model_path": "credit_risk_model.pkl",
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
}
```

## Feature Descriptions

| Feature | Description | Range/Type |
|---------|-------------|------------|
| `utilisation` | Credit utilization ratio | 0-100 (percentage) |
| `dpd_days` | Days past due on payments | 0-365 (days) |
| `cash_credit_ratio` | Ratio of cash credits to total credits | 0-1 (decimal) |
| `cash_debit_ratio` | Ratio of cash debits to total debits | 0-1 (decimal) |
| `inbound_cheque_bounce_count` | Number of inbound bounced cheques | Integer (count) |
| `inbound_cheque_bounce_amt` | Total amount of inbound bounced cheques | Float (amount) |
| `outbound_cheque_bounce_count` | Number of outbound bounced cheques | Integer (count) |
| `outbound_cheque_bounce_amt` | Total amount of outbound bounced cheques | Float (amount) |
| `total_amt_credit` | Total credit transactions amount | Float (amount) |
| `total_amt_debit` | Total debit transactions amount | Float (amount) |
| `no_of_banks` | Number of bank accounts | Integer (count) |

## Decision Logic

The API uses a four-tier decision system:

| Decision | Probability Range | Risk Level | Action |
|----------|------------------|------------|---------|
| **ACCEPT** | < 0.50 | Low | Grant credit with standard terms |
| **CONDITIONAL_ACCEPT** | 0.50 - 0.69 | Moderate | Approve with stricter terms |
| **REVIEW** | 0.70 - 0.79 | Elevated | Manual review required |
| **REJECT** | ≥ 0.80 | Critical | Deny credit immediately |

## Python Client Example

```python
import requests
import json

# API endpoint
API_URL = "http://localhost:5000"

# Example customer data
customer = {
    "customer_no": "CUST123456",
    "utilisation": 15.5,
    "dpd_days": 45,
    "cash_credit_ratio": 0.18,
    "cash_debit_ratio": 0.15,
    "inbound_cheque_bounce_count": 2,
    "inbound_cheque_bounce_amt": 8500,
    "outbound_cheque_bounce_count": 1,
    "outbound_cheque_bounce_amt": 3200,
    "total_amt_credit": 750000,
    "total_amt_debit": 680000,
    "no_of_banks": 4
}

# Score single customer
response = requests.post(
    f"{API_URL}/score",
    json=customer,
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    result = response.json()
    print(f"Customer: {result['customer_id']}")
    print(f"Risk Probability: {result['risk_score']['probability_percentage']}")
    print(f"Risk Bucket: {result['risk_score']['bucket']}")
    print(f"Decision: {result['decision']['status']}")
    print(f"Recommendation: {result['decision']['recommendation']}")
    print(f"\nExplanation: {result['decision']['explanation']}")
    print(f"\nTop Risk Drivers:")
    for driver in result['risk_drivers']:
        print(f"  - {driver['description']}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

## cURL Examples

### Score a customer
```bash
curl -X POST http://localhost:5000/score \
  -H "Content-Type: application/json" \
  -d '{
    "customer_no": "CUST123456",
    "utilisation": 15.5,
    "dpd_days": 45,
    "cash_credit_ratio": 0.18,
    "cash_debit_ratio": 0.15,
    "inbound_cheque_bounce_count": 2,
    "inbound_cheque_bounce_amt": 8500,
    "outbound_cheque_bounce_count": 1,
    "outbound_cheque_bounce_amt": 3200,
    "total_amt_credit": 750000,
    "total_amt_debit": 680000,
    "no_of_banks": 4
  }'
```

### Health check
```bash
curl http://localhost:5000/health
```

### Get model info
```bash
curl http://localhost:5000/model/info
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (missing/invalid data)
- `500`: Internal server error
- `503`: Service unavailable (model not loaded)

**Error Response Example:**
```json
{
  "error": "Missing required features: dpd_days, utilisation"
}
```

## Notes

- All numeric fields must be provided as numbers (not strings)
- Customer number (`customer_no`) is optional but recommended for tracking
- The API validates input ranges for key features
- SHAP values are used to explain which features drive the risk score
- Batch endpoint processes multiple customers efficiently
