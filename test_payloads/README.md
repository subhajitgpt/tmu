# Test Payloads for Credit Scoring API

This folder contains JSON payload files for testing the credit scoring API with various risk profiles.

## Individual Test Files

### Low Risk / Accept Scenarios

1. **excellent_profile.json** - Perfect credit profile
   - Expected: ACCEPT (Very low risk ~10-20%)
   - Zero DPD, low utilization, no bounced cheques

2. **accept_low_risk.json** - Good credit profile
   - Expected: ACCEPT (Low risk ~25-35%)
   - Minimal DPD, low utilization

3. **accept_moderate_low_risk.json** - Acceptable profile
   - Expected: ACCEPT (Moderate-low risk ~35-45%)
   - Some minor issues but overall safe

### Conditional Accept Scenarios

4. **conditional_accept_borderline.json** - Borderline acceptable
   - Expected: CONDITIONAL_ACCEPT (Risk ~50-60%)
   - Multiple warning signs, needs monitoring

5. **conditional_accept_higher_risk.json** - Higher conditional risk
   - Expected: CONDITIONAL_ACCEPT (Risk ~60-68%)
   - Significant issues but manageable with strict terms

### Review Required Scenarios

6. **review_elevated_risk.json** - Elevated risk requiring review
   - Expected: REVIEW (Risk ~70-75%)
   - High DPD, multiple bounced cheques

7. **review_high_risk.json** - High risk requiring careful review
   - Expected: REVIEW (Risk ~75-79%)
   - Very concerning indicators

### Reject Scenarios

8. **reject_high_risk.json** - High risk rejection
   - Expected: REJECT (Risk ~80-88%)
   - Exceeds acceptable thresholds

9. **reject_very_high_risk.json** - Critical risk rejection
   - Expected: REJECT (Risk ~88-95%)
   - Severe payment history issues

### Edge Cases

10. **edge_case_high_utilisation.json** - High utilization edge case
    - Tests behavior with maxed credit but otherwise good profile

11. **edge_case_many_banks.json** - Multiple bank accounts
    - Tests behavior with many banking relationships

### Batch Testing

12. **batch_mixed_profiles.json** - Mixed profiles batch test
    - Contains 5 customers covering all decision types
    - Tests batch endpoint functionality

## Testing Commands

### Test Individual Profile
```bash
# Windows PowerShell
curl.exe -X POST http://localhost:5000/score -H "Content-Type: application/json" -d (Get-Content test_payloads/accept_low_risk.json -Raw)

# Or using Invoke-RestMethod
$payload = Get-Content test_payloads/accept_low_risk.json | ConvertFrom-Json
Invoke-RestMethod -Uri http://localhost:5000/score -Method Post -Body ($payload | ConvertTo-Json) -ContentType "application/json"

# Linux/Mac
curl -X POST http://localhost:5000/score \
  -H "Content-Type: application/json" \
  -d @test_payloads/accept_low_risk.json
```

### Test Batch Processing
```bash
# Windows PowerShell
curl.exe -X POST http://localhost:5000/score/batch -H "Content-Type: application/json" -d (Get-Content test_payloads/batch_mixed_profiles.json -Raw)

# Linux/Mac
curl -X POST http://localhost:5000/score/batch \
  -H "Content-Type: application/json" \
  -d @test_payloads/batch_mixed_profiles.json
```

### Test All Files (PowerShell Script)
```powershell
# Test all individual payloads
$files = Get-ChildItem test_payloads/*.json -Exclude batch_*.json

foreach ($file in $files) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Testing: $($file.Name)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $payload = Get-Content $file.FullName | ConvertFrom-Json
    $response = Invoke-RestMethod -Uri http://localhost:5000/score `
                                  -Method Post `
                                  -Body ($payload | ConvertTo-Json) `
                                  -ContentType "application/json"
    
    Write-Host "Customer: $($response.customer_id)" -ForegroundColor Yellow
    Write-Host "Probability: $($response.risk_score.probability_percentage)" -ForegroundColor Yellow
    Write-Host "Bucket: $($response.risk_score.bucket)" -ForegroundColor Yellow
    Write-Host "Decision: $($response.decision.status)" -ForegroundColor $(
        switch ($response.decision.status) {
            "ACCEPT" { "Green" }
            "CONDITIONAL_ACCEPT" { "Yellow" }
            "REVIEW" { "Magenta" }
            "REJECT" { "Red" }
        }
    )
    Write-Host "Recommendation: $($response.decision.recommendation)"
}
```

## Expected Results Summary

| File | Expected Decision | Risk Range |
|------|------------------|------------|
| excellent_profile.json | ACCEPT | 10-20% |
| accept_low_risk.json | ACCEPT | 25-35% |
| accept_moderate_low_risk.json | ACCEPT | 35-45% |
| conditional_accept_borderline.json | CONDITIONAL_ACCEPT | 50-60% |
| conditional_accept_higher_risk.json | CONDITIONAL_ACCEPT | 60-68% |
| review_elevated_risk.json | REVIEW | 70-75% |
| review_high_risk.json | REVIEW | 75-79% |
| reject_high_risk.json | REJECT | 80-88% |
| reject_very_high_risk.json | REJECT | 88-95% |
| edge_case_high_utilisation.json | ACCEPT/CONDITIONAL | 40-55% |
| edge_case_many_banks.json | CONDITIONAL_ACCEPT | 55-65% |

## Python Test Script

```python
import requests
import json
from pathlib import Path

API_URL = "http://localhost:5000"

# Test all individual files
test_dir = Path("test_payloads")
for file_path in test_dir.glob("*.json"):
    if "batch" in file_path.name:
        continue
    
    print(f"\n{'='*60}")
    print(f"Testing: {file_path.name}")
    print('='*60)
    
    with open(file_path, 'r') as f:
        payload = json.load(f)
    
    response = requests.post(f"{API_URL}/score", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Customer: {result['customer_id']}")
        print(f"Probability: {result['risk_score']['probability_percentage']}")
        print(f"Bucket: {result['risk_score']['bucket']}")
        print(f"Decision: {result['decision']['status']}")
        print(f"Recommendation: {result['decision']['recommendation']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
```

## Notes

- All payloads use realistic values based on the feature distributions
- Customer IDs are descriptive to indicate expected outcome
- DPD (Days Past Due) is a strong predictor of risk
- Cheque bounce amounts and counts significantly impact decisions
- The model considers feature interactions, so results may vary slightly
- Batch file demonstrates processing multiple profiles efficiently
