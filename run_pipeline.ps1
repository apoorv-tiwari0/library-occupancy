# IIT Delhi Library Occupancy - Full Pipeline Deployment Script

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Launching Full Pipeline (Docker Storage + Zone Backend + Dashboard)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

if (Test-Path "venv\Scripts\python.exe") {
    & "venv\Scripts\python.exe" "setup_and_deploy.py"
} else {
    python setup_and_deploy.py
}
