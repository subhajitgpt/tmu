# Quick Start Guide - n8n Credit Scoring

Get the n8n credit scoring workflow running in under 10 minutes.

## Prerequisites

- ✅ Python 3.8+ installed
- ✅ Trained credit_risk_model.pkl exists (run `python train_and_save_model.py` if not)
- ✅ Docker installed (for n8n) OR npm installed

## Step 1: Start n8n (Choose One)

### Option A: Docker (Recommended)
```powershell
docker run -it --rm `
  --name n8n `
  -p 5678:5678 `
  -v ${PWD}/n8n-data:/home/node/.n8n `
  n8nio/n8n
```

### Option B: npm
```powershell
npm install n8n -g
n8n start
```

n8n will be available at: **http://localhost:5678**

## Step 2: Start ML Service

In a new terminal:

```powershell
# Navigate to n8n folder
cd n8n

# Install dependencies (if not already installed)
pip install flask catboost pandas

# Start the ML prediction service
python ml_service.py
```

You should see:
```
============================================================
ML Prediction Service for n8n Integration
============================================================
✓ Model loaded from ../credit_risk_model.pkl
✓ Service ready to accept predictions
✓ Listening on http://localhost:5064
============================================================
```

## Step 3: Import Workflow into n8n

1. Open browser: **http://localhost:5678**
2. Create account (first time only)
3. Click **"Workflows"** → **"Add workflow"**
4. Click **"..."** menu → **"Import from File"**
5. Select `credit_scoring_workflow.json` from this folder
6. Click **"Import"**

## Step 4: Configure (Optional)

### Enable Database Logging (Optional)
1. Click **"Log to PostgreSQL"** node
2. Click **"Credentials"** → **"Create New"**
3. Enter PostgreSQL details
4. Enable the node (remove "Disabled" flag)

### Enable Email Alerts (Optional)
1. Click **"Email Compliance Team"** node
2. Click **"Credentials"** → **"Create New"**
3. Enter SMTP details
4. Enable the node

### Enable Slack Notifications (Optional)
1. Click **"Slack Credit Officers"** node
2. Click **"Credentials"** → **"Add Slack Account"**
3. Connect your Slack workspace
4. Select channel
5. Enable the node

## Step 5: Activate Workflow

1. Click **"Active: OFF"** toggle at top right
2. Toggle should turn green: **"Active: ON"**
3. Copy the webhook URL (shown in Webhook node)

Example URL: `http://localhost:5678/webhook/credit-score`

## Step 6: Test It!

### Test Case 1: Low Risk (Accept)
```powershell
$body = @{
    customer_no = "CUST001"
    utilisation = 15.5
    dpd_days = 5
    cash_credit_ratio = 0.18
    cash_debit_ratio = 0.15
    inbound_cheque_bounce_count = 0
    inbound_cheque_bounce_amt = 0
    outbound_cheque_bounce_count = 0
    outbound_cheque_bounce_amt = 0
    total_amt_credit = 750000
    total_amt_debit = 680000
    no_of_banks = 2
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5678/webhook/credit-score" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"
```

### Test Case 2: High Risk (Reject)
```powershell
$body = @{
    customer_no = "CUST002"
    utilisation = 95.0
    dpd_days = 180
    cash_credit_ratio = 0.02
    cash_debit_ratio = 0.01
    inbound_cheque_bounce_count = 8
    inbound_cheque_bounce_amt = 45000
    outbound_cheque_bounce_count = 6
    outbound_cheque_bounce_amt = 38000
    total_amt_credit = 200000
    total_amt_debit = 850000
    no_of_banks = 8
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5678/webhook/credit-score" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"
```

### Use Test Payloads
```powershell
# From project root
$payload = Get-Content test_payloads/accept_low_risk.json | ConvertFrom-Json
Invoke-RestMethod -Uri "http://localhost:5678/webhook/credit-score" `
    -Method Post `
    -Body ($payload | ConvertTo-Json) `
    -ContentType "application/json"
```

## Expected Response

```json
{
  "customer_id": "CUST001",
  "timestamp": "2026-02-21T10:30:45.123456",
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
  "model_version": "1.0",
  "logged": true
}
```

## View Execution Logs

1. In n8n interface, click **"Executions"** tab
2. See all workflow runs with success/failure status
3. Click any execution to see detailed step-by-step flow
4. Debug issues by examining node outputs

## Common Issues

### Issue: "ML Service Connection Failed"
**Solution**: Ensure ML service is running
```powershell
# Check if service is running
curl http://localhost:5064/health
```

### Issue: "Missing required features"
**Solution**: Verify all 11 features are in the request body
```
utilisation, dpd_days, cash_credit_ratio, cash_debit_ratio,
inbound_cheque_bounce_count, inbound_cheque_bounce_amt,
outbound_cheque_bounce_count, outbound_cheque_bounce_amt,
total_amt_credit, total_amt_debit, no_of_banks
```

### Issue: "Workflow not triggering"
**Solution**: Check that:
1. Workflow is activated (toggle is green)
2. Webhook URL is correct
3. Request method is POST
4. Content-Type is application/json

## Next Steps

✅ **Customize Decision Logic**
- Adjust risk thresholds in "Calculate Risk Bucket" node
- Modify decision criteria in "Route By Decision" switch node

✅ **Add More Integrations**
- CRM updates (Salesforce, HubSpot)
- SMS notifications (Twilio)
- Data warehouse logging (BigQuery, Snowflake)
- Google Sheets audit trail

✅ **Schedule Batch Processing**
- Add Schedule Trigger node
- Process CSV files from SFTP/S3
- Generate daily reports

✅ **Deploy to Production**
- Use n8n Cloud (managed)
- Or deploy Docker container to cloud (AWS, Azure, GCP)
- Set up proper authentication on webhook
- Configure SSL/TLS certificates

## Architecture Overview

```
Customer Application
        ↓
   n8n Webhook
        ↓
   Validate Input ───→ If invalid, return error
        ↓
   HTTP → ML Service (localhost:5064/predict)
        ↓
   ML Service returns probability
        ↓
   Calculate Risk Bucket
        ↓
   Route by Decision (Switch)
   ├─ REJECT (prob ≥ 0.80)
   ├─ REVIEW (0.70 ≤ prob < 0.80)
   ├─ CONDITIONAL (0.50 ≤ prob < 0.70)
   └─ ACCEPT (prob < 0.50)
        ↓
   Format Response
        ↓
   Log to Database (optional)
        ↓
   Send Notifications (if high risk)
        ↓
   Return Response to Caller
```

## Resources

- **n8n Docs**: https://docs.n8n.io
- **Community Forum**: https://community.n8n.io
- **Workflow Library**: https://n8n.io/workflows
- **Video Tutorials**: https://www.youtube.com/@n8n-io

---

**Need Help?** Check the main [README.md](README.md) for detailed documentation.
