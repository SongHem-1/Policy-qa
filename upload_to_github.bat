@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   Policy QA System - Upload to GitHub
echo ============================================================
echo.

REM Check if git is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed. Please install Git first.
    echo Download: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo Step 1: Remove old git repo (if exists from parent folder)
if exist .git (
    echo Removing old .git folder...
    rmdir /s /q .git
    echo Done.
) else (
    echo No existing .git folder found, skipping.
)
echo.

echo Step 2: Initialize new Git repository
git init
git branch -M main
echo Done.
echo.

echo Step 3: Configure Git user info (if not set)
git config user.name >nul 2>nul
if %errorlevel% neq 0 (
    set /p git_name="Enter your GitHub username: "
    git config user.name "!git_name!"
)
git config user.email >nul 2>nul
if %errorlevel% neq 0 (
    set /p git_email="Enter your GitHub email: "
    git config user.email "!git_email!"
)
echo Done.
echo.

echo Step 4: Add all files (.gitignore will filter sensitive files)
git add .
echo Done.
echo.

echo ============================================================
echo   Checking staged files...
echo ============================================================
git status
echo.
echo ============================================================
echo   WARNING: Please verify the following are NOT in the list:
echo     - .env (API keys)
echo     - .venv/ (virtual environment)
echo     - chroma_db/ (vector database)
echo     - data/ (data files)
echo ============================================================
echo.

set /p confirm="Continue with commit? (y/n): "
if /i not "%confirm%"=="y" (
    echo Cancelled.
    pause
    exit /b 0
)

echo.
echo Step 5: Commit changes
git commit -m "Initial commit: Policy QA System with RAG, user auth, and memory management"
echo Done.
echo.

echo Step 6: Link to remote repository
set /p repo_url="Enter your GitHub repo URL (e.g. https://github.com/username/policy-qa.git): "

if "%repo_url%"=="" (
    echo No URL entered, skipping remote setup.
    echo You can add it later with:
    echo   git remote add origin YOUR_URL
    echo   git push -u origin main
    pause
    exit /b 0
)

git remote add origin %repo_url%
echo Done.
echo.

echo Step 7: Push to GitHub
echo.
echo NOTE: If this is your first time, you may need to enter your GitHub credentials.
echo Use your GitHub username and a Personal Access Token (NOT your password).
echo How to create a token: https://github.com/settings/tokens
echo.

git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ============================================================
    echo   SUCCESS! Project uploaded to GitHub!
    echo ============================================================
    echo.
    echo Repository URL: %repo_url%
    echo.
) else (
    echo.
    echo ============================================================
    echo   Push failed. Common solutions:
    echo ============================================================
    echo   1. Check your GitHub URL is correct
    echo   2. Make sure the repo exists on GitHub (empty, no README)
    echo   3. Use a Personal Access Token instead of password
    echo   4. If remote already exists, run:
    echo      git remote set-url origin YOUR_URL
    echo.
)

pause