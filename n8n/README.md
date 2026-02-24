# n8n Conversion for Credit Scoring API

## Overview

This folder contains resources for converting the Credit Scoring API to an n8n workflow-based architecture. The approach uses a **hybrid architecture** that leverages n8n's strengths in orchestration and integrations while maintaining Python for ML model inference.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   n8n Workflow                          │
├─────────────────────────────────────────────────────────┤
│ 1. Webhook receives credit application                 │
│ 2. Validate input data (required fields & ranges)      │
│ 3. HTTP Request to Python ML Service                   │
│ 4. Receive risk probability from model                 │
│ 5. Apply business logic (risk bucketing & decisions)   │
│ 6. Format standardized response                        │
│ 7. Log decision to database (PostgreSQL/MySQL)         │
│ 8. Send notifications based on decision:               │
│    - High risk → Email to compliance team              │
│    - Review needed → Slack message to officers         │
│ 9. Return webhook response to caller                   │
└────────────────────────────────┬────────────────────────┘
                                 │
                   ┌─────────────▼──────────────┐
                   │  Python ML Service          │
                   │  (Lightweight Flask/FastAPI)│
                   ├────────────────────────────┤
                   │ /predict endpoint           │
                   │ - Load CatBoost model       │
                   │ - Calculate risk probability│
                   │ - Compute SHAP values       │
                   │ - Return JSON response      │
                   └─────────────────────────────┘
```

## Why This Hybrid Approach?

### What n8n Handles Well ✅
- **Orchestration**: Coordinate multi-step workflows visually
- **Integrations**: 400+ pre-built connectors (Slack, Email, CRM, DBs)
- **Business Logic**: Risk bucketing, decision rules, routing
- **Notifications**: Multi-channel alerts (Email, SMS, Slack, Teams)
- **Data Logging**: Store decisions in databases or spreadsheets
- **Scheduling**: Batch processing, daily reports
- **No-code Changes**: Modify business rules without coding

### What Python ML Service Handles ❌
- **Model Inference**: CatBoost predictions require Python runtime
- **SHAP Values**: Feature importance calculation needs scientific libraries
- **NumPy/Pandas**: Matrix operations and data manipulation
- **Model Loading**: Pickle model deserialization

## Files in This Folder

- **README.md**: This documentation
- **credit_scoring_workflow.json**: Sample n8n workflow (import ready)
- **ml_service.py**: Lightweight Python service for model inference (to be created)

## Setup Instructions

### 1. Install n8n

**Option A: Self-Hosted (Docker)**
```bash
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n
```

**Option B: Self-Hosted (npm)**
```bash
npm install n8n -g
n8n start
```

**Option C: n8n Cloud**
Sign up at https://n8n.io

### 2. Import the Workflow

1. Open n8n at `http://localhost:5678`
2. Click **"Add workflow"** → **"Import from File"**
3. Select `credit_scoring_workflow.json`
4. The workflow will be imported with all nodes pre-configured

### 3. Start the ML Service

**Create a lightweight Flask service** (example in `ml_service.py`):

```python
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open('../credit_risk_model.pkl', 'rb') as f:
    MODEL = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = pd.DataFrame([data['features']])
    
    # Get prediction
    probability = float(MODEL.predict_proba(features)[:, 1][0])
    
    # Get SHAP values
    shap_values = MODEL.get_feature_importance(
        Pool(features), 
        type="ShapValues"
    )[0, :-1]
    
    return jsonify({
        "probability": probability,
        "shap_values": shap_values.tolist()
    })

if __name__ == '__main__':
    app.run(port=5064)  # Different port from main API
```

Run it:
```bash
python ml_service.py
```

### 4. Configure the Workflow

Update these settings in n8n workflow nodes:

**Webhook Node:**
- Method: POST
- Path: `/credit-score`
- Authentication: (optional) Basic Auth or Header Auth

**HTTP Request Node (ML Service):**
- URL: `http://localhost:5064/predict`
- Method: POST

**Database Nodes:**
- Configure your PostgreSQL/MySQL credentials
- Table: `credit_decisions`

**Notification Nodes:**
- Email: Configure SMTP settings
- Slack: Add Slack credentials

### 5. Test the Workflow

```powershell
# Test the n8n webhook
$body = @{
    customer_no = "CUST123456"
    utilisation = 15.5
    dpd_days = 45
    cash_credit_ratio = 0.18
    cash_debit_ratio = 0.15
    inbound_cheque_bounce_count = 2
    inbound_cheque_bounce_amt = 8500
    outbound_cheque_bounce_count = 1
    outbound_cheque_bounce_amt = 3200
    total_amt_credit = 750000
    total_amt_debit = 680000
    no_of_banks = 4
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5678/webhook/credit-score" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"
```

## Workflow Components (15 Nodes)

### Core Processing
1. **Webhook Trigger** - Receives POST requests
2. **Function: Validate Input** - Check required fields and ranges
3. **HTTP Request: ML Service** - Call Python prediction service
4. **Function: Calculate Risk Bucket** - Categorize probability into buckets
5. **Switch: Decision Logic** - Route based on risk level

### Decision Branches (4 paths)
6. **Set: ACCEPT Response** - probability < 0.50
7. **Set: CONDITIONAL_ACCEPT Response** - 0.50 ≤ probability < 0.70
8. **Set: REVIEW Response** - 0.70 ≤ probability < 0.80
9. **Set: REJECT Response** - probability ≥ 0.80

### Logging & Notifications
10. **PostgreSQL: Log Decision** - Store all decisions
11. **IF: High Risk Check** - probability ≥ 0.70
12. **Email: Compliance Alert** - For high-risk cases
13. **Slack: Review Notification** - For manual review cases
14. **Merge: Combine Branches** - Merge all decision paths
15. **Respond to Webhook** - Return final response

## Sample Response

```json
{
  "customer_id": "CUST123456",
  "timestamp": "2026-02-21T08:30:45.123456",
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
  "logged": true,
  "notification_sent": false
}
```

## Benefits of n8n Approach

### 1. **Visual Workflow Management**
- See entire credit decision process at a glance
- Non-technical users can understand the flow
- Easy to identify bottlenecks

### 2. **No-Code Business Logic Changes**
- Adjust risk thresholds without touching code
- Modify decision criteria in minutes
- Test changes in staging environment

### 3. **Rich Integrations**
- **Salesforce/HubSpot**: Auto-update CRM records
- **Google Sheets**: Export daily decisions for analysis
- **Slack/Teams**: Real-time alerts for high-risk cases
- **Email**: Automated customer communications
- **SMS**: Send application status via Twilio
- **Zapier**: Connect to 5000+ additional apps

### 4. **Batch Processing**
- Schedule daily batch scoring jobs
- Process uploaded CSV files automatically
- Generate end-of-day reports

### 5. **Audit & Compliance**
- Log every decision with timestamp
- Track who modified business rules
- Generate compliance reports easily

### 6. **A/B Testing**
- Route 10% of traffic to new model version
- Compare decision outcomes
- Gradual rollout of changes

## Advanced Use Cases

### 1. Multi-Channel Application Intake
```
Email → Parse Attachment → Validate → n8n Workflow → Send Response Email
Web Form → Webhook → Validate → n8n Workflow → Update CRM
Mobile App → API → Validate → n8n Workflow → Push Notification
```

### 2. Scheduled Batch Processing
```
Schedule Trigger (Daily 2 AM)
   ↓
Read CSV from SFTP
   ↓
Split Into Batches (100 records)
   ↓
For Each: Call ML Service
   ↓
Aggregate Results
   ↓
Write to Database
   ↓
Send Summary Email to Management
```

### 3. Human-in-the-Loop Review
```
Risk Score = REVIEW
   ↓
Create Jira Ticket
   ↓
Send Slack Message to Credit Officer
   ↓
Wait for Manual Decision (Webhook)
   ↓
Update Customer Record
   ↓
Send Decision Email
```

## Performance Considerations

| Metric | Current Flask API | n8n Hybrid |
|--------|------------------|------------|
| **Latency** | ~50-100ms | ~150-300ms |
| **Throughput** | 100+ req/sec | 20-50 req/sec |
| **Scalability** | Vertical (single process) | Horizontal (queue-based) |
| **Best For** | High-frequency scoring | Workflow orchestration |

**Recommendation**: Use n8n for workflows with notifications/integrations. Keep Flask API for high-volume raw scoring.

## Cost Analysis

### Self-Hosted n8n (Free)
- ✅ Unlimited workflows
- ✅ Unlimited executions
- ❌ Self-manage infrastructure
- ❌ Manual updates

### n8n Cloud (Starting $20/month)
- ✅ Managed infrastructure
- ✅ Automatic updates
- ✅ Better uptime
- ❌ Execution limits on lower tiers

## Migration Checklist

- [ ] Install n8n (Docker/npm/Cloud)
- [ ] Import workflow JSON
- [ ] Create lightweight ML service (Flask/FastAPI)
- [ ] Configure database credentials
- [ ] Set up email/Slack integrations
- [ ] Test with sample payloads
- [ ] Configure production webhook URLs
- [ ] Set up monitoring/alerts
- [ ] Train team on n8n interface
- [ ] Document custom nodes/functions

## Troubleshooting

### Issue: "ML Service Connection Failed"
**Solution**: Ensure Python service is running on correct port
```bash
curl http://localhost:5064/predict -X POST
```

### Issue: "Validation Failed"
**Solution**: Check Function node for required field names
- Ensure all 11 features are present
- Verify data types (numeric values)

### Issue: "Database Insert Failed"
**Solution**: Verify PostgreSQL credentials and table exists
```sql
CREATE TABLE credit_decisions (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    timestamp TIMESTAMP,
    probability DECIMAL(5,4),
    decision VARCHAR(20),
    logged_at TIMESTAMP DEFAULT NOW()
);
```

## Next Steps

1. **Review the workflow JSON** - Understand the node structure
2. **Customize decision logic** - Adjust risk thresholds for your business
3. **Add your integrations** - Connect to your CRM, email, database
4. **Test thoroughly** - Use test payloads from `../test_payloads/`
5. **Deploy to production** - Use n8n Cloud or Docker deployment

## Support & Resources

- **n8n Documentation**: https://docs.n8n.io
- **n8n Community**: https://community.n8n.io
- **Example Workflows**: https://n8n.io/workflows
- **Video Tutorials**: https://www.youtube.com/@n8n-io

## License

This workflow is provided as-is for educational purposes. Modify as needed for your production environment.

---

**Created**: February 21, 2026  
**Version**: 1.0  
**Author**: Credit Scoring API Team
