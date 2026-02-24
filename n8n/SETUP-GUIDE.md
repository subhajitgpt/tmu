# 🚀 AUTOMATED n8n SETUP - STEP BY STEP GUIDE

## 📋 Complete Setup in 5 Simple Steps

This guide will walk you through setting up the n8n credit scoring workflow with **fully automated scripts**. No manual configuration needed!

---

## ✅ Prerequisites

Before you begin, ensure you have:
- ✅ Windows 10 or later
- ✅ PowerShell (comes with Windows)
- ✅ Trained model file: `credit_risk_model.pkl` (located in parent directory)
- ✅ Python 3.8+ with required packages (Flask, CatBoost, Pandas)

**Check Python installation:**
```powershell
python --version
```

If Python is not installed, download from: https://www.python.org/downloads/

---

## 🎯 5-Step Automated Setup

### **STEP 1: Install Node.js**

Node.js is required to run n8n.

```powershell
.\1-install-nodejs.ps1
```

**What this script does:**
- ✓ Checks if Node.js is already installed
- ✓ If not, opens the Node.js download page
- ✓ Provides installation instructions
- ✓ Verifies installation after setup

**Follow the prompts:**
1. Script will open https://nodejs.org/en/download/ in your browser
2. Download the **LTS (Long Term Support) version** for Windows
3. Run the installer with default settings
4. ✨ **IMPORTANT**: Restart PowerShell after installation
5. Run the script again to verify: `.\1-install-nodejs.ps1`

**Expected output after successful installation:**
```
✓ Node.js is already installed!
  Node.js version: v20.x.x
  npm version: 10.x.x
```

---

### **STEP 2: Install n8n**

Install n8n globally on your system.

```powershell
.\2-install-n8n.ps1
```

**What this script does:**
- ✓ Verifies Node.js is installed
- ✓ Checks if n8n is already installed
- ✓ Installs n8n globally via npm
- ✓ Confirms successful installation

**Duration:** 2-5 minutes (depending on internet speed)

**Expected output:**
```
✓ Node.js detected: v20.x.x
✓ npm detected: 10.x.x
Installing n8n globally...
✓ n8n installed successfully!
Installed version: v1.x.x
```

---

### **STEP 3: Start All Services**

This is the **magic script** that starts everything you need!

```powershell
.\3-start-services.ps1
```

**What this script does:**
- ✓ Verifies all prerequisites (Node.js, n8n, model file)
- ✓ Starts **ML Service** in a new window (port 5064)
- ✓ Waits for ML service to be healthy
- ✓ Starts **n8n** in a new window (port 5678)
- ✓ Opens n8n in your browser automatically

**You will see 3 windows:**
1. **This window** - Confirmation messages
2. **ML Service window** - Python service running
3. **n8n window** - n8n server running

**Expected output:**
```
✓ All prerequisites met!
✓ ML Service starting in new window (port 5064)
✓ ML Service is healthy!
✓ n8n starting in new window (port 5678)

📊 ML Service: http://localhost:5064
🔧 n8n Interface: http://localhost:5678
```

**Services running:**
- 🐍 ML Service: http://localhost:5064
- 🔧 n8n: http://localhost:5678

---

### **STEP 4: Import Workflow**

Import the pre-built credit scoring workflow into n8n.

```powershell
.\4-import-workflow.ps1
```

**What this script does:**
- ✓ Verifies workflow file exists
- ✓ Checks if n8n is running
- ✓ Opens n8n in your browser
- ✓ Opens file explorer to the workflow file
- ✓ Provides step-by-step import instructions

**Follow the on-screen instructions:**

1. **Create n8n account (first time only):**
   - Enter your email and password
   - This is stored locally on your machine (not cloud)

2. **Import the workflow:**
   - Click **"Workflows"** in left menu
   - Click **"+ Add workflow"** button
   - Click **"..."** (three dots menu)
   - Select **"Import from File"**
   - Choose `credit_scoring_workflow.json` (file explorer will open to this location)
   - Click **"Import"**

3. **Activate the workflow:**
   - Click **"Active: OFF"** toggle at top-right
   - Should turn green showing **"Active: ON"**

4. **Note the webhook URL:**
   - Click the **"Webhook - Credit Application"** node
   - Copy the webhook URL (usually: `http://localhost:5678/webhook/credit-score`)

**Screenshot of successful import:**
- Workflow will show 15 connected nodes
- Green "Active: ON" indicator
- Webhook URL displayed in the Webhook node

---

### **STEP 5: Test the Workflow**

Automatically test your workflow with sample data.

```powershell
.\5-test-workflow.ps1
```

**What this script does:**
- ✓ Verifies both ML Service and n8n are running
- ✓ Runs 3 automated test cases:
  - **Test 1**: Low risk application (Expected: ACCEPT)
  - **Test 2**: High risk application (Expected: REJECT)
  - **Test 3**: Moderate risk application (Expected: REVIEW)
- ✓ Displays results for each test
- ✓ Confirms workflow is working correctly

**Expected output:**
```
✓ ML Service is healthy
✓ n8n is running

Test 1: Low Risk Application (Expected: ACCEPT)
-------------------------------------------
Customer ID: TEST_001
Risk Score: 35.21%
Risk Bucket: No Risk
Decision: ACCEPT
Recommendation: Application approved

Test 2: High Risk Application (Expected: REJECT)
-------------------------------------------
Customer ID: TEST_002
Risk Score: 92.45%
Risk Bucket: Very High
Decision: REJECT
Recommendation: Application rejected due to high credit risk

Test 3: Moderate Risk Application (Expected: REVIEW)
-------------------------------------------
Customer ID: TEST_003
Risk Score: 73.18%
Risk Bucket: Moderate
Decision: REVIEW
Recommendation: Requires manual review by credit officer

✓ Testing Complete!
Your n8n credit scoring workflow is working!
```

---

## 🎉 Setup Complete!

Congratulations! Your n8n credit scoring workflow is now fully operational.

---

## 💡 What You Can Do Now

### 1. **View Execution Logs in n8n**
```
1. Open http://localhost:5678
2. Click "Executions" tab
3. See detailed logs of all workflow runs
4. Click any execution to see step-by-step flow
```

### 2. **Test with Custom Data**
```powershell
$customData = @{
    customer_no = "CUST_12345"
    utilisation = 25.0
    dpd_days = 10
    cash_credit_ratio = 0.15
    cash_debit_ratio = 0.12
    inbound_cheque_bounce_count = 1
    inbound_cheque_bounce_amt = 5000
    outbound_cheque_bounce_count = 0
    outbound_cheque_bounce_amt = 0
    total_amt_credit = 500000
    total_amt_debit = 450000
    no_of_banks = 3
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5678/webhook/credit-score" `
    -Method Post `
    -Body $customData `
    -ContentType "application/json"
```

### 3. **Use Test Payloads from Parent Directory**
```powershell
# Navigate to parent directory
cd ..

# Test with pre-built payloads
$payload = Get-Content test_payloads/accept_low_risk.json
Invoke-RestMethod -Uri "http://localhost:5678/webhook/credit-score" `
    -Method Post `
    -Body $payload `
    -ContentType "application/json"
```

### 4. **Customize the Workflow**

In n8n interface:

**Change Risk Thresholds:**
- Edit "Route By Decision" node
- Adjust probability cutoffs (currently: 0.50, 0.70, 0.80)

**Add Notifications:**
- Enable "Email Compliance Team" node
- Configure SMTP credentials
- Enable "Slack Credit Officers" node
- Add Slack workspace credentials

**Add Database Logging:**
- Enable "Log to PostgreSQL" node
- Configure PostgreSQL connection
- Create table: `credit_decisions`

**Add More Integrations:**
- Drag new nodes from left panel
- Available: Salesforce, HubSpot, Google Sheets, Twilio, etc.

---

## 📊 System Architecture

```
Customer Application (API Request)
        ↓
n8n Webhook (http://localhost:5678/webhook/credit-score)
        ↓
Validate Input (Check all 11 required features)
        ↓
HTTP Request → ML Service (http://localhost:5064/predict)
        ↓
ML Service (Python + CatBoost)
├── Load model
├── Calculate probability
└── Return SHAP values
        ↓
Calculate Risk Bucket (Based on probability)
        ↓
Route By Decision (Switch node)
├── REJECT (probability ≥ 0.80)
├── REVIEW (0.70 ≤ probability < 0.80)
├── CONDITIONAL_ACCEPT (0.50 ≤ probability < 0.70)
└── ACCEPT (probability < 0.50)
        ↓
Format Response (Add decision details)
        ↓
Optional: Log to Database
Optional: Send Notifications
        ↓
Return Response to Caller
```

---

## 🔧 Running the Services Daily

### **Every Time You Want to Use the System:**

**Option A: Run Master Setup (Recommended)**
```powershell
.\SETUP-ALL.ps1
```
This automatically runs steps 3-5 for you!

**Option B: Manual Start**
```powershell
# Start services
.\3-start-services.ps1

# Then test
.\5-test-workflow.ps1
```

### **Stopping the Services:**
Simply close the ML Service and n8n PowerShell windows.

---

## 🛠️ Troubleshooting

### **Issue: "Node.js is not installed"**
**Solution:**
```powershell
.\1-install-nodejs.ps1
# Follow the prompts to install Node.js
# Restart PowerShell after installation
```

### **Issue: "n8n is not installed"**
**Solution:**
```powershell
.\2-install-n8n.ps1
```

### **Issue: "Model file not found"**
**Solution:**
```powershell
cd ..  # Go to parent directory
python train_and_save_model.py
cd n8n
```

### **Issue: "ML Service Connection Failed"**
**Solution:**
```powershell
# Check if ML service is running
curl http://localhost:5064/health

# If not, start services again
.\3-start-services.ps1
```

### **Issue: "Workflow test failed"**
**Solution:**
1. Ensure workflow is imported in n8n
2. Ensure workflow is activated (green toggle)
3. Verify webhook URL: `http://localhost:5678/webhook/credit-score`

### **Issue: "Permission denied when installing"**
**Solution:**
Run PowerShell as Administrator:
1. Right-click PowerShell
2. Select "Run as Administrator"
3. Run the install scripts again

---

## 📁 Files in This Directory

| File | Purpose |
|------|---------|
| `1-install-nodejs.ps1` | Guides Node.js installation |
| `2-install-n8n.ps1` | Installs n8n globally |
| `3-start-services.ps1` | Starts ML Service and n8n |
| `4-import-workflow.ps1` | Guides workflow import |
| `5-test-workflow.ps1` | Tests the workflow |
| `SETUP-ALL.ps1` | Master script (runs 3-5 automatically) |
| `credit_scoring_workflow.json` | n8n workflow definition |
| `ml_service.py` | Lightweight ML prediction service |
| `README.md` | Technical documentation |
| `SETUP-GUIDE.md` | This step-by-step guide |

---

## 🌟 Advanced Features

### **Enable Email Notifications**
1. Open workflow in n8n
2. Click "Email Compliance Team" node
3. Click "Credentials" → "Create New"
4. Enter SMTP details (Gmail, Office365, etc.)
5. Remove "Disabled" flag from node
6. Save workflow

### **Enable Slack Notifications**
1. Click "Slack Credit Officers" node
2. Click "Credentials" → "Add Slack account"
3. Connect your Slack workspace
4. Select target channel
5. Remove "Disabled" flag
6. Save workflow

### **Enable Database Logging**
1. Install PostgreSQL or use existing database
2. Create table:
```sql
CREATE TABLE credit_decisions (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    probability DECIMAL(5,4),
    bucket VARCHAR(20),
    decision_status VARCHAR(30),
    timestamp TIMESTAMP,
    logged_at TIMESTAMP DEFAULT NOW()
);
```
3. In n8n, click "Log to PostgreSQL" node
4. Add PostgreSQL credentials
5. Remove "Disabled" flag
6. Save workflow

### **Schedule Batch Processing**
1. In n8n, add "Schedule Trigger" node
2. Configure: Daily at 2 AM
3. Add "Read Binary File" node (read CSV)
4. Add "Split In Batches" node
5. Connect to existing workflow
6. Save workflow

---

## 📞 Support & Resources

- **n8n Documentation**: https://docs.n8n.io
- **n8n Community Forum**: https://community.n8n.io
- **n8n Workflow Library**: https://n8n.io/workflows
- **Video Tutorials**: https://www.youtube.com/@n8n-io
- **CatBoost Documentation**: https://catboost.ai/docs

---

## 🎓 Next Steps

**Week 1: Get Familiar**
- Run tests with different data
- Explore n8n execution logs
- Customize decision thresholds

**Week 2: Add Integrations**
- Enable email/Slack notifications
- Connect to your database
- Add CRM integration (Salesforce/HubSpot)

**Week 3: Production Ready**
- Set up proper authentication on webhooks
- Deploy to cloud (AWS/Azure/GCP)
- Configure SSL/TLS certificates
- Set up monitoring and alerts

---

## ✅ Quick Reference Card

| Task | Command |
|------|---------|
| Install Node.js | `.\1-install-nodejs.ps1` |
| Install n8n | `.\2-install-n8n.ps1` |
| Start services | `.\3-start-services.ps1` |
| Import workflow | `.\4-import-workflow.ps1` |
| Test workflow | `.\5-test-workflow.ps1` |
| Full setup | `.\SETUP-ALL.ps1` |
| Open n8n | `http://localhost:5678` |
| Check ML service | `http://localhost:5064/health` |
| Webhook URL | `http://localhost:5678/webhook/credit-score` |

---

**Created**: February 21, 2026  
**Version**: 1.0  
**Author**: Credit Scoring API Team

🎉 **Enjoy your automated credit scoring workflow!**
