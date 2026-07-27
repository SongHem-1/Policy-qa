@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo ============================================================
echo   查看评估报告
echo ============================================================
echo.

python view_evaluation_report.py

echo.
pause