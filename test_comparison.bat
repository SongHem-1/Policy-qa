@echo off
cd /d "%~dp0"

echo ============================================================
echo   检索策略对比测试
echo   固定权重混合检索 vs 自适应检索（查询分类 + 查询扩展）
echo ============================================================
echo.

REM 激活虚拟环境
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo Running test...
python test_retrieval_comparison.py

echo.
echo ============================================================
echo Test completed! Check results in:
echo   data/retrieval_comparison.json
echo ============================================================
pause