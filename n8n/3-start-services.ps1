# ========================================
# Step 3: Start All Services
# ========================================
# This script starts both n8n and the ML service

param(
    [switch]$SkipN8n
)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Starting Credit Scoring Services" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Locate a usable Python interpreter (prefer workspace venv)
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

$pythonCmd = $null
if (Test-Path $venvPython) {
    $pythonCmd = $venvPython
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} else {
    Write-Host "✗ Python not found!" -ForegroundColor Red
    Write-Host "Install Python 3.10+ or create a venv at: $venvPython" -ForegroundColor Yellow
    exit 1
}

# Check if Node.js is installed
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "✗ Node.js is not installed!" -ForegroundColor Red
    Write-Host "Run: .\1-install-nodejs.ps1" -ForegroundColor Yellow
    exit 1
}

# Check if n8n is installed
if (-not (Get-Command n8n -ErrorAction SilentlyContinue)) {
    Write-Host "✗ n8n is not installed!" -ForegroundColor Red
    Write-Host "Run: .\2-install-n8n.ps1" -ForegroundColor Yellow
    exit 1
}

# Check if model file exists
if (-not (Test-Path "..\credit_risk_model.pkl")) {
    Write-Host "✗ Model file not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please train the model first:" -ForegroundColor Yellow
    Write-Host "  cd .." -ForegroundColor Gray
    Write-Host "  $pythonCmd train_and_save_model.py" -ForegroundColor Gray
    Write-Host "  cd n8n" -ForegroundColor Gray
    exit 1
}

Write-Host "✓ All prerequisites met!" -ForegroundColor Green
Write-Host ""

# Start ML Service in a new window
Write-Host "Starting ML Service..." -ForegroundColor Yellow
$mlServicePath = Join-Path $PSScriptRoot "ml_service.py"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; Write-Host 'Starting ML Service...' -ForegroundColor Cyan; & '$pythonCmd' ml_service.py" -WindowStyle Normal

Write-Host "✓ ML Service starting in new window (port 5064)" -ForegroundColor Green
Start-Sleep -Seconds 3

# Test if ML service is running
try {
    Write-Host "Checking ML Service health..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
    $response = Invoke-RestMethod -Uri "http://localhost:5064/health" -Method Get -TimeoutSec 5
    if ($response.status -eq "healthy") {
        Write-Host "✓ ML Service is healthy!" -ForegroundColor Green
    }
}
catch {
    Write-Host "⚠ ML Service may still be starting up..." -ForegroundColor Yellow
    Write-Host "  Check the ML Service window for status" -ForegroundColor Gray
}

Write-Host ""

# Start n8n in a new window
if (-not $SkipN8n) {
    Write-Host "Starting n8n..." -ForegroundColor Yellow
    Write-Host "This will open in a new window..." -ForegroundColor Gray
    Write-Host ""
    
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Starting n8n...' -ForegroundColor Cyan; Write-Host 'Access n8n at: http://localhost:5678' -ForegroundColor Green; Write-Host ''; n8n" -WindowStyle Normal
    
    Write-Host "✓ n8n starting in new window (port 5678)" -ForegroundColor Green
    Start-Sleep -Seconds 5
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Services Started!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 ML Service: http://localhost:5064" -ForegroundColor White
Write-Host "🔧 n8n Interface: http://localhost:5678" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Open browser: http://localhost:5678" -ForegroundColor Yellow
Write-Host "2. Create n8n account (first time only)" -ForegroundColor Yellow
Write-Host "3. Import workflow: credit_scoring_workflow.json" -ForegroundColor Yellow
Write-Host ""
Write-Host "Or run the automated import:" -ForegroundColor Cyan
Write-Host "  .\4-import-workflow.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to open n8n in browser..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Start-Process "http://localhost:5678"
