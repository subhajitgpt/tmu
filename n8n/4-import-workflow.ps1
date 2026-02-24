# ========================================
# Step 4: Import Workflow Guide
# ========================================
# This script provides step-by-step guidance to import the workflow

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Import n8n Workflow" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if workflow file exists
$workflowFile = Join-Path $PSScriptRoot "credit_scoring_workflow.json"
if (-not (Test-Path $workflowFile)) {
    Write-Host "✗ Workflow file not found!" -ForegroundColor Red
    Write-Host "Expected: $workflowFile" -ForegroundColor Gray
    exit 1
}

Write-Host "✓ Workflow file found: credit_scoring_workflow.json" -ForegroundColor Green
Write-Host ""

# Try to check if n8n is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5678" -Method Get -TimeoutSec 2 -UseBasicParsing
    Write-Host "✓ n8n is running at http://localhost:5678" -ForegroundColor Green
}
catch {
    Write-Host "⚠ n8n doesn't appear to be running" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please start n8n first:" -ForegroundColor Yellow
    Write-Host "  .\3-start-services.ps1" -ForegroundColor Gray
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        exit 0
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Import Instructions" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Follow these steps to import the workflow:" -ForegroundColor White
Write-Host ""
Write-Host "1. Open n8n in your browser:" -ForegroundColor Cyan
Write-Host "   http://localhost:5678" -ForegroundColor Green
Write-Host ""
Write-Host "2. Create account (first time only):" -ForegroundColor Cyan
Write-Host "   - Enter your email and password" -ForegroundColor Gray
Write-Host "   - This is stored locally on your machine" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Import the workflow:" -ForegroundColor Cyan
Write-Host "   a. Click 'Workflows' in the left menu" -ForegroundColor Gray
Write-Host "   b. Click '+ Add workflow' button" -ForegroundColor Gray
Write-Host "   c. Click the '...' (three dots) menu" -ForegroundColor Gray
Write-Host "   d. Select 'Import from File'" -ForegroundColor Gray
Write-Host "   e. Browse to: $PSScriptRoot" -ForegroundColor Gray
Write-Host "   f. Select: credit_scoring_workflow.json" -ForegroundColor Gray
Write-Host "   g. Click 'Import'" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Activate the workflow:" -ForegroundColor Cyan
Write-Host "   - Click the 'Active: OFF' toggle at top-right" -ForegroundColor Gray
Write-Host "   - It should turn green showing 'Active: ON'" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Note the webhook URL:" -ForegroundColor Cyan
Write-Host "   - Click the 'Webhook - Credit Application' node" -ForegroundColor Gray
Write-Host "   - Copy the webhook URL (usually: http://localhost:5678/webhook/credit-score)" -ForegroundColor Gray
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Open n8n in browser
Write-Host "Opening n8n in your browser..." -ForegroundColor Yellow
Start-Process "http://localhost:5678"
Write-Host ""

# Open file explorer to the workflow file
Write-Host "Opening file explorer to workflow location..." -ForegroundColor Yellow
Start-Process explorer.exe -ArgumentList "/select,`"$workflowFile`""
Write-Host ""

Write-Host "After importing, run the test script:" -ForegroundColor Cyan
Write-Host "  .\5-test-workflow.ps1" -ForegroundColor Yellow
Write-Host ""
