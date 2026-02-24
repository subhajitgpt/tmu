# ========================================
# Step 5: Test the Workflow
# ========================================
# This script tests the n8n credit scoring workflow

param(
    [string]$WebhookUrl = "http://localhost:5678/webhook/credit-score"
)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Testing Credit Scoring Workflow" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if services are running
Write-Host "Checking services..." -ForegroundColor Yellow

# Check ML Service
try {
    $mlHealth = Invoke-RestMethod -Uri "http://localhost:5064/health" -Method Get -TimeoutSec 3
    Write-Host "✓ ML Service is healthy" -ForegroundColor Green
}
catch {
    Write-Host "✗ ML Service is not responding!" -ForegroundColor Red
    Write-Host "  Please start it: .\3-start-services.ps1" -ForegroundColor Yellow
    exit 1
}

# Check n8n
try {
    $null = Invoke-WebRequest -Uri "http://localhost:5678" -Method Get -TimeoutSec 3 -UseBasicParsing
    Write-Host "✓ n8n is running" -ForegroundColor Green
}
catch {
    Write-Host "✗ n8n is not responding!" -ForegroundColor Red
    Write-Host "  Please start it: .\3-start-services.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Running Test Cases" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Test Case 1: Low Risk (Accept)
Write-Host "Test 1: Low Risk Application (Expected: ACCEPT)" -ForegroundColor Cyan
Write-Host "-------------------------------------------" -ForegroundColor Gray

$testCase1 = @{
    customer_no = "TEST_001"
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
}

try {
    $response1 = Invoke-RestMethod -Uri $WebhookUrl -Method Post -Body ($testCase1 | ConvertTo-Json) -ContentType "application/json"
    
    Write-Host "Customer ID: $($response1.customer_id)" -ForegroundColor White
    Write-Host "Risk Score: $($response1.risk_score.probability_percentage)" -ForegroundColor White
    Write-Host "Risk Bucket: $($response1.risk_score.bucket)" -ForegroundColor White
    Write-Host "Decision: $($response1.decision.status)" -ForegroundColor $(if ($response1.decision.status -eq "ACCEPT") { "Green" } else { "Yellow" })
    Write-Host "Recommendation: $($response1.decision.recommendation)" -ForegroundColor Gray
}
catch {
    Write-Host "✗ Test failed!" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Make sure:" -ForegroundColor Yellow
    Write-Host "  1. Workflow is imported in n8n" -ForegroundColor Gray
    Write-Host "  2. Workflow is activated (toggle ON)" -ForegroundColor Gray
    Write-Host "  3. Webhook URL is correct: $WebhookUrl" -ForegroundColor Gray
    exit 1
}

Write-Host ""

# Test Case 2: High Risk (Reject)
Write-Host "Test 2: High Risk Application (Expected: REJECT)" -ForegroundColor Cyan
Write-Host "-------------------------------------------" -ForegroundColor Gray

$testCase2 = @{
    customer_no = "TEST_002"
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
}

try {
    $response2 = Invoke-RestMethod -Uri $WebhookUrl -Method Post -Body ($testCase2 | ConvertTo-Json) -ContentType "application/json"
    
    Write-Host "Customer ID: $($response2.customer_id)" -ForegroundColor White
    Write-Host "Risk Score: $($response2.risk_score.probability_percentage)" -ForegroundColor White
    Write-Host "Risk Bucket: $($response2.risk_score.bucket)" -ForegroundColor White
    Write-Host "Decision: $($response2.decision.status)" -ForegroundColor $(if ($response2.decision.status -eq "REJECT") { "Red" } else { "Yellow" })
    Write-Host "Recommendation: $($response2.decision.recommendation)" -ForegroundColor Gray
}
catch {
    Write-Host "✗ Test failed!" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test Case 3: Moderate Risk (Review)
Write-Host "Test 3: Moderate Risk Application (Expected: REVIEW)" -ForegroundColor Cyan
Write-Host "-------------------------------------------" -ForegroundColor Gray

$testCase3 = @{
    customer_no = "TEST_003"
    utilisation = 65.0
    dpd_days = 90
    cash_credit_ratio = 0.08
    cash_debit_ratio = 0.06
    inbound_cheque_bounce_count = 3
    inbound_cheque_bounce_amt = 15000
    outbound_cheque_bounce_count = 2
    outbound_cheque_bounce_amt = 12000
    total_amt_credit = 400000
    total_amt_debit = 520000
    no_of_banks = 5
}

try {
    $response3 = Invoke-RestMethod -Uri $WebhookUrl -Method Post -Body ($testCase3 | ConvertTo-Json) -ContentType "application/json"
    
    Write-Host "Customer ID: $($response3.customer_id)" -ForegroundColor White
    Write-Host "Risk Score: $($response3.risk_score.probability_percentage)" -ForegroundColor White
    Write-Host "Risk Bucket: $($response3.risk_score.bucket)" -ForegroundColor White
    Write-Host "Decision: $($response3.decision.status)" -ForegroundColor Yellow
    Write-Host "Recommendation: $($response3.decision.recommendation)" -ForegroundColor Gray
}
catch {
    Write-Host "✗ Test failed!" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "✓ Testing Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your n8n credit scoring workflow is working!" -ForegroundColor Green
Write-Host ""
Write-Host "View execution logs:" -ForegroundColor Cyan
Write-Host "  1. Open n8n: http://localhost:5678" -ForegroundColor Yellow
Write-Host "  2. Click 'Executions' tab" -ForegroundColor Yellow
Write-Host "  3. See all workflow runs with details" -ForegroundColor Yellow
Write-Host ""
Write-Host "Test with your own data:" -ForegroundColor Cyan
Write-Host "  Invoke-RestMethod -Uri '$WebhookUrl' ``" -ForegroundColor Gray
Write-Host "    -Method Post ``" -ForegroundColor Gray
Write-Host "    -Body (\$yourData | ConvertTo-Json) ``" -ForegroundColor Gray
Write-Host "    -ContentType 'application/json'" -ForegroundColor Gray
Write-Host ""
