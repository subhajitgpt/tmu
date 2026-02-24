# ========================================
# MASTER SETUP SCRIPT
# ========================================
# This script runs the complete setup process automatically
# after Node.js and n8n are installed

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AUTOMATED n8n SETUP" -ForegroundColor Cyan
Write-Host "  Credit Scoring Workflow" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if this is first run
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "🚀 FIRST TIME SETUP" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "This appears to be your first time running the setup." -ForegroundColor White
    Write-Host "Please follow these steps:" -ForegroundColor White
    Write-Host ""
    Write-Host "STEP 1: Install Node.js" -ForegroundColor Cyan
    Write-Host "  .\1-install-nodejs.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "STEP 2: Install n8n" -ForegroundColor Cyan
    Write-Host "  .\2-install-n8n.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "STEP 3: Run this script again" -ForegroundColor Cyan
    Write-Host "  .\SETUP-ALL.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    $response = Read-Host "Start Node.js installation now? (y/n)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        & "$PSScriptRoot\1-install-nodejs.ps1"
    }
    exit 0
}

if (-not (Get-Command n8n -ErrorAction SilentlyContinue)) {
    Write-Host "📦 n8n NOT INSTALLED" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Node.js is installed, but n8n is not." -ForegroundColor White
    Write-Host ""
    Write-Host "Installing n8n now..." -ForegroundColor Cyan
    Write-Host ""
    
    & "$PSScriptRoot\2-install-n8n.ps1"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "✗ n8n installation failed" -ForegroundColor Red
        Write-Host "Please run manually: .\2-install-n8n.ps1" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host ""
    Write-Host "✓ n8n installed! Continuing with setup..." -ForegroundColor Green
    Write-Host ""
    Start-Sleep -Seconds 2
}

# All prerequisites met, run the full setup
Write-Host "✓ Prerequisites met!" -ForegroundColor Green
Write-Host "  - Node.js installed" -ForegroundColor Gray
Write-Host "  - n8n installed" -ForegroundColor Gray
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  STARTING AUTOMATED SETUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Start services
Write-Host "STEP 1: Starting services..." -ForegroundColor Cyan
Write-Host ""
& "$PSScriptRoot\3-start-services.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ Failed to start services" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Waiting for services to fully start..." -ForegroundColor Yellow
Start-Sleep -Seconds 8

# Step 2: Import workflow guide
Write-Host ""
Write-Host "STEP 2: Import workflow..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Opening import guide..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

& "$PSScriptRoot\4-import-workflow.ps1"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  WORKFLOW IMPORT" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Please complete the workflow import in your browser:" -ForegroundColor White
Write-Host ""
Write-Host "1. Create n8n account (if first time)" -ForegroundColor Gray
Write-Host "2. Import the workflow file" -ForegroundColor Gray
Write-Host "3. Activate the workflow (toggle ON)" -ForegroundColor Gray
Write-Host ""
Read-Host "Press ENTER when workflow is imported and activated"

# Step 3: Test workflow
Write-Host ""
Write-Host "STEP 3: Testing workflow..." -ForegroundColor Cyan
Write-Host ""
Start-Sleep -Seconds 2

& "$PSScriptRoot\5-test-workflow.ps1"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE! 🎉" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your n8n credit scoring workflow is ready!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Services Running:" -ForegroundColor Cyan
Write-Host "  - ML Service: http://localhost:5064" -ForegroundColor White
Write-Host "  - n8n Interface: http://localhost:5678" -ForegroundColor White
Write-Host ""
Write-Host "📖 Documentation:" -ForegroundColor Cyan
Write-Host "  - Full guide: SETUP-GUIDE.md" -ForegroundColor White
Write-Host "  - Technical docs: README.md" -ForegroundColor White
Write-Host "  - Quick start: QUICKSTART.md" -ForegroundColor White
Write-Host ""
Write-Host "🧪 Test Webhook:" -ForegroundColor Cyan
Write-Host "  http://localhost:5678/webhook/credit-score" -ForegroundColor White
Write-Host ""
Write-Host "To restart services in the future:" -ForegroundColor Cyan
Write-Host "  .\3-start-services.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "To run tests again:" -ForegroundColor Cyan
Write-Host "  .\5-test-workflow.ps1" -ForegroundColor Yellow
Write-Host ""
