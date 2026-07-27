@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   Policy QA System - Quick Start
echo ============================================================
echo.

echo Select startup mode:
echo   1. Start API Service
echo   2. Start Web Interface with User Login
echo   3. Start Full System - API and Web with User Login
echo.

set /p choice="Please select (1/2/3): "

if "%choice%"=="1" (
    echo.
    echo Starting API Service...
    call run_api.bat
) else if "%choice%"=="2" (
    echo.
    echo Starting Web Interface with User Login...
    call run_web.bat
) else if "%choice%"=="3" (
    echo.
    echo Starting API Service...
    start "API Server" cmd /c run_api.bat

    echo Waiting for API service to start...
    timeout /t 10 /nobreak >nul

    echo Starting Web Interface with User Login...
    call run_web.bat
) else (
    echo Invalid selection, exiting
    pause
    exit /b 1
)

pause