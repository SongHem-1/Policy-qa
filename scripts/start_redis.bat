@echo off
REM 启动本地 Redis（Windows 便携版），供会话存储与 RQ 任务队列使用
REM 用法：双击运行，或在终端执行 scripts\start_redis.bat
REM 说明：必须由交互式终端启动（本 Agent 沙箱环境禁止脱离前台的进程监听端口）

set "REDIS_SERVER=%USERPROFILE%\tools\redis8\Redis-8.10.0-Windows-x64-msys2\redis-server.exe"

if not exist "%REDIS_SERVER%" (
  echo [ERROR] 未找到 redis-server.exe
  echo 请从 https://github.com/redis-windows/redis-windows/releases 下载
  echo Redis-8.10.0-Windows-x64-msys2.zip 并解压到 %USERPROFILE%\tools\redis8\
  exit /b 1
)

tasklist /fi "imagename eq redis-server.exe" 2>nul | find "redis-server.exe" >nul
if %errorlevel%==0 (
  echo Redis 已在运行（127.0.0.1:6379）
  exit /b 0
)

start "PolicyQA-Redis" "%REDIS_SERVER%" --port 6379 --bind 127.0.0.1
timeout /t 1 /nobreak >nul
echo Redis 已启动: 127.0.0.1:6379
