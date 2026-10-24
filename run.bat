@echo off
echo ========================================================
echo   FraudShield - frontend + backend on http://127.0.0.1:8000
echo ========================================================
cd /d "%~dp0"
start http://127.0.0.1:8000
python backend\app.py
pause
