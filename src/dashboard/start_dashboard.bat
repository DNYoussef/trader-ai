@echo off
REM Gary×Taleb Risk Dashboard Startup Script for Windows
REM This script starts both the WebSocket server and frontend development server

setlocal enabledelayedexpansion

echo 🚀 Starting Gary×Taleb Risk Dashboard...

REM Function to check if a port is in use
:check_port
netstat -an | findstr ":%1 " >nul 2>&1
if %errorlevel% == 0 (
    exit /b 0
) else (
    exit /b 1
)

REM Check prerequisites
echo 🔍 Checking prerequisites...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed or not in PATH
    pause
    exit /b 1
)

npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm is not installed or not in PATH
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed

REM Start backend server
echo 📡 Starting WebSocket server...

REM Check if port 8000 is already in use
call :check_port 8000
if %errorlevel% == 0 (
    echo ❌ Port 8000 is already in use. Please stop the existing service.
    pause
    exit /b 1
)

REM Create logs directory
if not exist logs mkdir logs

REM Install Python dependencies if needed
if not exist .backend_deps_installed (
    echo 📦 Installing Python dependencies...
    pip install -r server\requirements.txt
    echo. > .backend_deps_installed
)

REM Start server in background
echo ⏳ Starting WebSocket server...
start /B python run_server.py > logs\server.log 2>&1

REM Wait for server to start
set /a counter=0
:wait_backend
timeout /t 1 /nobreak >nul
call :check_port 8000
if %errorlevel% == 0 (
    echo ✅ WebSocket server started on http://localhost:8000
    goto start_frontend
)
set /a counter+=1
if %counter% lss 30 goto wait_backend

echo ❌ Failed to start WebSocket server
pause
exit /b 1

:start_frontend
echo 🎨 Starting frontend development server...

REM Check if port 3000 is already in use
call :check_port 3000
if %errorlevel% == 0 (
    echo ❌ Port 3000 is already in use. Please stop the existing service.
    pause
    exit /b 1
)

cd frontend

REM Install Node dependencies if needed
if not exist node_modules (
    echo 📦 Installing Node.js dependencies...
    npm install
)

REM Start frontend development server
echo ⏳ Starting Vite development server...
start /B npm run dev

cd ..

REM Wait for frontend to start
set /a counter=0
:wait_frontend
timeout /t 1 /nobreak >nul
call :check_port 3000
if %errorlevel% == 0 (
    echo ✅ Frontend development server started on http://localhost:3000
    goto dashboard_ready
)
set /a counter+=1
if %counter% lss 30 goto wait_frontend

echo ❌ Failed to start frontend development server
pause
exit /b 1

:dashboard_ready
echo.
echo 🎉 Dashboard is ready!
echo 📊 Frontend: http://localhost:3000
echo 📡 Backend API: http://localhost:8000
echo 🔌 WebSocket: ws://localhost:8000/ws
echo.
echo 📋 Useful commands:
echo   • View server logs: type logs\server.log
echo   • Health check: curl http://localhost:8000/api/health
echo   • Open dashboard: start http://localhost:3000
echo.
echo 🔄 Dashboard is running. Press any key to stop and exit.

pause >nul

REM Cleanup - kill background processes
echo 🛑 Shutting down dashboard...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM node.exe /T >nul 2>&1
echo ✅ Dashboard stopped

endlocal