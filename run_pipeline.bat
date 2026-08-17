@echo off
TITLE IIT Delhi Library Occupancy - Pipeline Deployment
echo ======================================================================
echo Launching Production Pipeline (Docker Storage + Zone Backend + React Dashboard)
echo ======================================================================

IF EXIST venv\Scripts\python.exe (
    venv\Scripts\python.exe setup_and_deploy.py
) ELSE (
    python setup_and_deploy.py
)

pause
