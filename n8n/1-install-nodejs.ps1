# ========================================
# Step 1: Install Node.js
# ========================================
# This script helps you install Node.js which is required for n8n

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Node.js Installation Guide" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if Node.js is already installed
if (Get-Command node -ErrorAction SilentlyContinue) {
    $nodeVersion = node --version
    $npmVersion = npm --version
    Write-Host "✓ Node.js is already installed!" -ForegroundColor Green
    Write-Host "  Node.js version: $nodeVersion" -ForegroundColor Green
    Write-Host "  npm version: $npmVersion" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can proceed to the next step!" -ForegroundColor Green
    Write-Host "Run: .\2-install-n8n.ps1" -ForegroundColor Yellow
    exit 0
}

Write-Host "Node.js is not installed on your system." -ForegroundColor Yellow
Write-Host ""
Write-Host "Please follow these steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Open your browser and go to:" -ForegroundColor White
Write-Host "   https://nodejs.org/en/download/" -ForegroundColor Green
Write-Host ""
Write-Host "2. Download the LTS (Long Term Support) version for Windows" -ForegroundColor White
Write-Host ""
Write-Host "3. Run the installer and follow the installation wizard" -ForegroundColor White
Write-Host "   - Accept the license agreement" -ForegroundColor Gray
Write-Host "   - Use default installation settings" -ForegroundColor Gray
Write-Host "   - Make sure 'Add to PATH' is checked" -ForegroundColor Gray
Write-Host ""
Write-Host "4. After installation, RESTART this PowerShell window" -ForegroundColor White
Write-Host ""
Write-Host "5. Run this script again to verify installation:" -ForegroundColor White
Write-Host "   .\1-install-nodejs.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan

# Offer to open the download page
$response = Read-Host "Would you like to open the Node.js download page now? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Start-Process "https://nodejs.org/en/download/"
    Write-Host ""
    Write-Host "✓ Opening Node.js download page in your browser..." -ForegroundColor Green
    Write-Host ""
    Write-Host "After installation, restart PowerShell and run:" -ForegroundColor Yellow
    Write-Host "  .\1-install-nodejs.ps1" -ForegroundColor Yellow
}
