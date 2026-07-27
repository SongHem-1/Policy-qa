@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ========================================
echo   国家政策知识库智能问答系统
echo   （带用户认证和对话历史）
echo ========================================
echo.

cd /d "%~dp0"

if not exist .venv\Scripts\activate.bat (
    echo 错误：未找到虚拟环境
    echo 请先运行以下命令创建虚拟环境：
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo 正在启动 API 服务...
echo API服务将初始化知识库（包括OCR处理）
start "Policy QA API" cmd /c "python api.py & pause"
timeout /t 5 /nobreak >nul

echo 正在启动 Gradio Web 服务...
echo Web界面将直接使用API服务，无需重复OCR处理
echo 启动后请在浏览器访问：http://127.0.0.1:7862
echo.

python app.py

pause