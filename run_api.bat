@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo ============================================
echo   Policy QA API Server
echo   FastAPI + Uvicorn
echo ============================================
echo.
call .venv\Scripts\activate.bat
echo 启动 API 服务 (http://127.0.0.1:8000) ...
echo API 文档: http://127.0.0.1:8000/docs
echo.
python api.py
pause