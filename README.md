# Credit Risk Scoring System

A complete machine learning-powered credit risk assessment system with a RESTful API built using CatBoost and Flask.

## 🎯 Overview

This system provides real-time credit risk scoring for loan applications using a trained CatBoost classifier. It analyzes 11 key financial features to predict default probability and provides automated accept/reject decisions with detailed explanations.

## 📁 Project Structure

```
TMU/
├── tmu_v1.0.py - tmu_v1.4.py    # Evolution of training interface
├── credit_scoring_api.py         # Flask API for credit scoring
├── train_and_save_model.py       # Model training and persistence
├── credit_risk_model.pkl         # Trained model (generated)
├── data.csv                      # Training data
├── requirements.txt              # Python dependencies
├── API_USAGE_GUIDE.md           # Detailed API documentation
├── test_payloads/               # Test JSON payloads
│   ├── README.md               # Testing guide
│   ├── accept_*.json           # Low risk profiles
│   ├── conditional_*.json      # Moderate risk profiles
│   ├── review_*.json           # High risk profiles
│   ├── reject_*.json           # Critical risk profiles
│   ├── edge_case_*.json        # Edge cases
│   └── batch_*.json            # Batch testing
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation

1. **Install dependencies**
```powershell
pip install -r requirements.txt
```

2. **Train the model**
```powershell
python train_and_save_model.py
```
This will:
- Generate 50,000 training samples
- Train a CatBoost classifier
- Save the model as `credit_risk_model.pkl`
- Display performance metrics

3. **Start the API server**
```powershell
python credit_scoring_api.py
```
The API will be available at `http://localhost:5000`

4. **Test the API**
```powershell
# Health check
curl http://localhost:5000/health

# Score a customer
$payload = Get-Content test_payloads\accept_low_risk.json | ConvertFrom-Json
Invoke-RestMethod -Uri http://localhost:5000/score -Method Post -Body ($payload | ConvertTo-Json) -ContentType "application/json"
```

## 🔑 Key Features

### Machine Learning
- **Algorithm**: CatBoost Classifier with robust hyperparameters
- **Features**: 11 financial indicators
- **Training**: Automated train/validation/test split
- **Evaluation**: AUC-ROC, precision, recall metrics
- **Explainability**: SHAP values for feature importance

### API Capabilities
- ✅ Single customer scoring with detailed decision
- ✅ Batch processing for multiple customers
- ✅ Risk probability calculation (0-100%)
- ✅ Risk bucket classification (Very High, High, Moderate, Low, No Risk)
- ✅ Automated accept/reject decisions
- ✅ Top 3 risk drivers with explanations
- ✅ Health monitoring and model info endpoints

### Decision Framework

| Decision | Risk Probability | Action |
|----------|-----------------|---------|
| **ACCEPT** | < 50% | Grant credit with standard terms |
| **CONDITIONAL_ACCEPT** | 50-69% | Approve with stricter terms |
| **REVIEW** | 70-79% | Manual review required |
| **REJECT** | ≥ 80% | Deny credit immediately |

## 📊 Input Features

| Feature | Description | Type |
|---------|-------------|------|
| `utilisation` | Credit utilization ratio (%) | Float (0-100) |
| `dpd_days` | Days past due on payments | Integer (0-365) |
| `cash_credit_ratio` | Cash credits / total credits | Float (0-1) |
| `cash_debit_ratio` | Cash debits / total debits | Float (0-1) |
| `inbound_cheque_bounce_count` | Number of inbound bounced cheques | Integer |
| `inbound_cheque_bounce_amt` | Amount of inbound bounced cheques | Float |
| `outbound_cheque_bounce_count` | Number of outbound bounced cheques | Integer |
| `outbound_cheque_bounce_amt` | Amount of outbound bounced cheques | Float |
| `total_amt_credit` | Total credit transactions | Float |
| `total_amt_debit` | Total debit transactions | Float |
| `no_of_banks` | Number of bank accounts | Integer |

## 🔌 API Endpoints

### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "credit_risk_model.pkl"
}
```

### `POST /score`
Score a single customer application

**Request:**
```json
{
  "customer_no": "CUST001",
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

**Response:**
```json
{
  "customer_id": "CUST001",
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
    "explanation": "Customer shows a 35.2% probability of default..."
  },
  "risk_drivers": [
    {
      "feature": "dpd_days",
      "value": 45.0,
      "impact": 0.0234,
      "description": "dpd_days (45.00) increases risk by 0.0234"
    }
  ],
  "model_version": "1.0"
}
```

### `POST /score/batch`
Score multiple customers in one request

**Request:**
```json
{
  "customers": [
    { /* customer 1 data */ },
    { /* customer 2 data */ }
  ]
}
```

### `GET /model/info`
Get model metadata and decision thresholds

## 🧪 Testing

### Run All Tests
```powershell
# Test all individual profiles
Get-ChildItem test_payloads\*.json -Exclude batch_*.json | ForEach-Object {
    Write-Host "`nTesting: $($_.Name)" -ForegroundColor Cyan
    $payload = Get-Content $_.FullName | ConvertFrom-Json
    Invoke-RestMethod -Uri http://localhost:5000/score -Method Post -Body ($payload | ConvertTo-Json) -ContentType "application/json"
}
```

### Test Categories
- **Accept profiles**: Low risk, should be approved
- **Conditional profiles**: Moderate risk, approve with conditions
- **Review profiles**: High risk, manual review needed
- **Reject profiles**: Critical risk, should be denied
- **Edge cases**: Boundary conditions and unusual patterns
- **Batch**: Multiple customers at once

See [test_payloads/README.md](test_payloads/README.md) for detailed testing instructions.

## 📈 Model Performance

The model is trained with:
- **Training set**: 75% of data (60% train, 15% validation)
- **Test set**: 25% of data
- **Early stopping**: 50 rounds without improvement
- **Regularization**: L2 regularization, subsampling, column sampling
- **Iterations**: Up to 600 (with early stopping)

Expected performance:
- **AUC-ROC**: ~0.85-0.90
- **Precision** (at top 30%): ~0.75-0.85
- **Recall** (at top 30%): ~0.70-0.80

## 🛠️ Development

### Training Interface Evolution
- `tmu_v1.0.py` - Initial training interface
- `tmu_v1.1.py` - Added validation
- `tmu_v1.2.py` - Enhanced metrics
- `tmu_v1.3.py` - Added Flask UI
- `tmu_v1.4.py` - Full featured web interface

### Model Retraining
```powershell
# Retrain with new data
python train_and_save_model.py

# The API will automatically use the new model on restart
python credit_scoring_api.py
```

### Customizing Decision Thresholds

Edit `credit_scoring_api.py` to adjust decision boundaries:

```python
def get_decision(probability: float, bucket: str) -> Dict[str, Any]:
    if probability >= 0.80:  # Adjust threshold
        return {"status": "REJECT", ...}
    # ... other thresholds
```

## 📚 Documentation

- **[API_USAGE_GUIDE.md](API_USAGE_GUIDE.md)** - Comprehensive API documentation with examples
- **[test_payloads/README.md](test_payloads/README.md)** - Testing guide and payload descriptions

## 🔒 Security Notes

- Change `app.secret_key` in production
- Add authentication/authorization for production use
- Validate and sanitize all inputs
- Use HTTPS in production
- Implement rate limiting
- Add request logging and monitoring

## 🐛 Troubleshooting

### Model Not Found Error
```
⚠ Warning: Model file credit_risk_model.pkl not found
```
**Solution**: Run `python train_and_save_model.py` first

### Port Already in Use
```
Address already in use
```
**Solution**: Change port in `credit_scoring_api.py` or kill the process:
```powershell
# Find process on port 5000
netstat -ano | findstr :5000
# Kill process
taskkill /PID <PID> /F
```

### Module Not Found
```
ModuleNotFoundError: No module named 'catboost'
```
**Solution**: Install dependencies:
```powershell
pip install -r requirements.txt
```

### Invalid Input Error
```json
{"error": "Missing required features: utilisation"}
```
**Solution**: Ensure all 11 features are present in the payload

## 📝 Requirements

```
catboost>=1.2
Flask>=2.3.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

## 🤝 Contributing

1. Follow existing code structure
2. Add tests for new features
3. Update documentation
4. Maintain backward compatibility

## 📄 License

This project is for educational and demonstration purposes.

## 🎓 Use Cases

- **Banks**: Automated loan application screening
- **Fintech**: Real-time credit decisioning
- **Lending Platforms**: Risk assessment for peer-to-peer lending
- **Credit Unions**: Member loan evaluation
- **Financial Services**: Portfolio risk management

## 🔮 Future Enhancements

- [ ] Add authentication and user management
- [ ] Implement model versioning and A/B testing
- [ ] Add real-time monitoring and alerting
- [ ] Create web UI for manual reviews
- [ ] Add model retraining pipeline
- [ ] Implement feature drift detection
- [ ] Add database integration for logging
- [ ] Create Docker containerization
- [ ] Add comprehensive unit tests
- [ ] Implement CI/CD pipeline

## 📞 Support

For issues, questions, or contributions, please refer to the documentation files in this repository.

---

**Built with**: Python, CatBoost, Flask, scikit-learn, pandas, numpy
