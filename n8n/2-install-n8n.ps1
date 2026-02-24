# ========================================
# Step 2: Install n8n
# ========================================
# This script installs n8n globally using npm

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Installing n8n" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if Node.js is installed
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "✗ Node.js is not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run: .\1-install-nodejs.ps1 first" -ForegroundColor Yellow
    exit 1
}

$nodeVersion = node --version
$npmVersion = npm --version
Write-Host "✓ Node.js detected: $nodeVersion" -ForegroundColor Green
Write-Host "✓ npm detected: $npmVersion" -ForegroundColor Green
Write-Host ""

# Check if n8n is already installed
if (Get-Command n8n -ErrorAction SilentlyContinue) {
    $n8nVersion = n8n --version
    Write-Host "✓ n8n is already installed: v$n8nVersion" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can proceed to the next step!" -ForegroundColor Green
    Write-Host "Run: .\3-start-services.ps1" -ForegroundColor Yellow
    exit 0
}

# Install n8n
Write-Host "Installing n8n globally..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
Write-Host ""

try {
    npm install n8n -g
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "✓ n8n installed successfully!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
    
    $n8nVersion = n8n --version
    Write-Host "Installed version: v$n8nVersion" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next step:" -ForegroundColor Cyan
    Write-Host "Run: .\3-start-services.ps1" -ForegroundColor Yellow
    Write-Host ""
}
catch {
    Write-Host ""
    Write-Host "✗ Installation failed!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Try running PowerShell as Administrator and retry" -ForegroundColor Yellow
    exit 1
}
