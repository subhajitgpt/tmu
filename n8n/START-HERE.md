# 🎯 START HERE - Complete n8n Setup

## 📌 What You Have

✅ **Model file exists**: `credit_risk_model.pkl`  
✅ **Automated scripts created**: 11 files ready to use  
✅ **Complete workflow**: Ready to import into n8n

---

## 🚀 QUICK START (5 Steps)

### If Node.js is already installed:

```powershell
# Run the master setup script
.\SETUP-ALL.ps1
```

That's it! The script will:
1. ✓ Verify Node.js and n8n
2. ✓ Start ML Service (port 5064)
3. ✓ Start n8n (port 5678)
4. ✓ Guide you through import
5. ✓ Run automated tests

---

### If Node.js is NOT installed (First Time):

Run these commands **one at a time**:

```powershell
# Step 1: Install Node.js
.\1-install-nodejs.ps1
# (Follow prompts, install Node.js, RESTART PowerShell)

# Step 2: Install n8n
.\2-install-n8n.ps1
# (Will take 2-5 minutes)

# Step 3: Start everything
.\3-start-services.ps1
# (Opens 2 new windows + browser)

# Step 4: Import workflow
.\4-import-workflow.ps1
# (Follow on-screen instructions)

# Step 5: Test it
.\5-test-workflow.ps1
# (Runs 3 automated tests)
```

---

## 📁 What's in This Folder

| File | What It Does |
|------|--------------|
| **🎯 SETUP-ALL.ps1** | **Master script - runs everything** |
| 1-install-nodejs.ps1 | Guides Node.js installation |
| 2-install-n8n.ps1 | Installs n8n |
| 3-start-services.ps1 | Starts ML Service + n8n |
| 4-import-workflow.ps1 | Guides workflow import |
| 5-test-workflow.ps1 | Tests the workflow |
| credit_scoring_workflow.json | The n8n workflow (import this) |
| ml_service.py | Python ML prediction service |
| **📖 SETUP-GUIDE.md** | **Detailed step-by-step guide** |
| README.md | Technical documentation |
| QUICKSTART.md | 10-minute quick start |

---

## ⏱️ Time Required

- **First time setup**: 15-20 minutes
- **Daily usage**: 2 minutes (just run `.\3-start-services.ps1`)

---

## 🎯 Your Action Now

### **Option A: Quick Setup (Recommended)**
```powershell
.\SETUP-ALL.ps1
```

### **Option B: Step-by-Step**
Read **SETUP-GUIDE.md** for detailed instructions

---

## 📞 Need Help?

### Common Issues

**"Node.js is not installed"**
→ Run `.\1-install-nodejs.ps1`

**"n8n is not installed"**
→ Run `.\2-install-n8n.ps1`

**"Model file not found"**
→ Train model: `cd .. && python train_and_save_model.py && cd n8n`

**Tests fail**
→ Make sure workflow is imported and activated in n8n

---

## ✅ Success Looks Like

After setup, you'll have:

✅ ML Service running on http://localhost:5064  
✅ n8n running on http://localhost:5678  
✅ Workflow imported and activated  
✅ All tests passing  
✅ Webhook URL ready to use: `http://localhost:5678/webhook/credit-score`

---

## 🎓 Next Steps After Setup

1. **View execution logs** in n8n (http://localhost:5678)
2. **Test with your data** using the webhook
3. **Customize** decision thresholds in workflow
4. **Add integrations** (Email, Slack, Database)

---

## 🔄 Daily Usage

```powershell
# Start services
.\3-start-services.ps1

# Test
.\5-test-workflow.ps1

# Stop: Just close the ML and n8n windows
```

---

**Ready? Pick an option above and let's go! 🚀**

Need detailed help? Open **SETUP-GUIDE.md**
