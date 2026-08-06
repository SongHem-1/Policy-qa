@echo off
REM Start local Redis (Windows portable build) for sessions and RQ queue.
REM Usage: double-click this file, or run: scripts\start_redis.bat
REM NOTE: Must be started from an interactive terminal.

set "REDIS_SERVER=%USERPROFILE%\tools\redis8\Redis-8.10.0-Windows-x64-msys2\redis-server.exe"

if not exist "%REDIS_SERVER%" (
  echo [ERROR] redis-server.exe not found at:
  echo   %REDIS_SERVER%
  echo.
  echo Please download Redis-8.10.0-Windows-x64-msys2.zip from
  echo https://github.com/redis-windows/redis-windows/releases
  echo and extract it to %USERPROFILE%\tools\redis8\
  pause
  exit /b 1
)

tasklist /fi "imagename eq redis-server.exe" 2>nul | find "redis-server.exe" >nul
if %errorlevel%==0 (
  echo Redis is already running on 127.0.0.1:6379
  exit /b 0
)

start "PolicyQA-Redis" "%REDIS_SERVER%" --port 6379 --bind 127.0.0.1
ping -n 2 127.0.0.1 >nul

tasklist /fi "imagename eq redis-server.exe" 2>nul | find "redis-server.exe" >nul
if %errorlevel%==0 (
  echo Redis started: 127.0.0.1:6379
) else (
  echo [ERROR] redis-server exited immediately. Check for port conflicts or run it manually:
  echo   "%REDIS_SERVER%" --port 6379 --bind 127.0.0.1
  pause
  exit /b 1
)
