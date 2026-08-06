@echo off
REM Start local Redis (Windows portable build) for sessions and RQ queue.
REM Usage: double-click this file, or run: scripts\start_redis.bat
REM NOTE: Must be started from an interactive terminal.
REM Port 6379 is inside the Windows excluded port range (6299-6398, Hyper-V
REM reservation) on this machine, so the default is 16379.
REM Override: scripts\start_redis.bat 6399

set "REDIS_SERVER=%USERPROFILE%\tools\redis8\Redis-8.10.0-Windows-x64-msys2\redis-server.exe"
set "REDIS_LOG=%USERPROFILE%\tools\redis8\redis-start.log"
set "REDIS_PORT=16379"
if not "%1"=="" set "REDIS_PORT=%1"

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
  echo Redis is already running
  exit /b 0
)

echo Starting Redis on port %REDIS_PORT% ...
start "PolicyQA-Redis" "%REDIS_SERVER%" --port %REDIS_PORT% --logfile "%REDIS_LOG%"
ping -n 3 127.0.0.1 >nul

tasklist /fi "imagename eq redis-server.exe" 2>nul | find "redis-server.exe" >nul
if %errorlevel%==0 (
  echo Redis started: 127.0.0.1:%REDIS_PORT%
) else (
  echo.
  echo [ERROR] redis-server exited immediately. Last log lines:
  if exist "%REDIS_LOG%" (
    type "%REDIS_LOG%"
  ) else (
    echo   [no log file was produced]
  )
  echo.
  echo Try running it in the foreground to see the exact error:
  echo   "%REDIS_SERVER%" --port %REDIS_PORT%
  echo.
  echo If the port is occupied, check: netstat -ano ^| findstr :%REDIS_PORT%
  echo.
  echo NOTE: port 6379 is inside the Windows excluded range 6299-6398,
  echo so use 16379 by default or another free port.
  pause
  exit /b 1
)
