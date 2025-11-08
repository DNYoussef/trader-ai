@echo off
REM ========================================
REM UNIFIED STARTUP SCRIPT - Gary×Taleb Trading System
REM Starts both backend and frontend servers with one command
REM ========================================

setlocal enabledelayedexpansion

echo ========================================
echo   GARY×TALEB TRADING SYSTEM LAUNCHER
echo   Unified Dashboard + Trading Engine
echo ========================================
echo.

REM Colors disabled for compatibility
set "GREEN="
set "YELLOW="
set "RED="
set "RESET="

REM Function to check if a command exists
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

where node >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: npm is not installed or not in PATH
    echo Please install Node.js which includes npm
    pause
    exit /b 1
)

echo [OK] Prerequisites verified
echo.

REM Check if ports are already in use
netstat -an | findstr ":8000" >nul 2>&1
if %errorlevel% == 0 (
    echo WARNING: Port 8000 is already in use
    echo Either the backend is already running or another service is using this port.
    echo.
    choice /M "Do you want to continue anyway"
    if !errorlevel! neq 1 exit /b 0
)

netstat -an | findstr ":3000" >nul 2>&1
if %errorlevel% == 0 (
    echo WARNING: Port 3000 is already in use
    echo Either the frontend is already running or another service is using this port.
    echo.
    choice /M "Do you want to continue anyway"
    if !errorlevel! neq 1 exit /b 0
)

REM ========================================
REM BACKEND SERVER STARTUP
REM ========================================
echo [1/3] Starting Backend Server (FastAPI + WebSocket)...

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Using virtual environment...
    call venv\Scripts\activate.bat
)

REM Install/update Python dependencies if needed
if not exist ".backend_installed" (
    echo Installing Python dependencies...
    pip install -r requirements.txt >nul 2>&1
    if %errorlevel% neq 0 (
        echo Failed to install Python dependencies
        echo Run manually: pip install -r requirements.txt
    ) else (
        echo. > .backend_installed
        echo [OK] Python dependencies installed
    )
) else (
    echo [OK] Python dependencies already installed
)

REM Start backend server in new window
echo Launching backend server...
start "Trading Backend - Port 8000" cmd /k "cd /d src\dashboard && python run_server_simple.py"

REM Wait for backend to initialize
echo Waiting for backend to start...
timeout /t 3 /nobreak >nul

REM ========================================
REM FRONTEND SERVER STARTUP
REM ========================================
echo.
echo [2/3] Starting Frontend Server (React + Vite)...

REM Check if node_modules exists and install if needed
cd src\dashboard\frontend
if not exist "node_modules" (
    echo Installing frontend dependencies (this may take a few minutes)...
    call npm install
    if %errorlevel% neq 0 (
        echo Failed to install frontend dependencies
        echo Run manually: cd src\dashboard\frontend && npm install
        cd ..\..\..
        pause
        exit /b 1
    )
    echo [OK] Frontend dependencies installed
) else (
    echo [OK] Frontend dependencies already installed
)

REM Start frontend in new window
echo Launching frontend server...
start "Trading Frontend - Port 3000" cmd /k "npm run dev"
cd ..\..\..

REM Wait for frontend to compile
echo Waiting for frontend to compile...
timeout /t 5 /nobreak >nul

REM ========================================
REM FINAL STATUS
REM ========================================
echo.
echo [3/3] Opening Dashboard...
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo    SYSTEM STARTED SUCCESSFULLY!
echo ========================================
echo.
echo Frontend Dashboard: http://localhost:3000
echo Backend API: http://localhost:8000
echo WebSocket: ws://localhost:8000/ws/{client_id}
echo API Documentation: http://localhost:8000/docs
echo.
echo SERVICES RUNNING:
echo   • Backend Server (FastAPI) - Window: "Trading Backend"
echo   • Frontend Server (Vite) - Window: "Trading Frontend"
echo.
echo USEFUL COMMANDS:
echo   • View this system: http://localhost:3000
echo   • Check API health: curl http://localhost:8000/api/health
echo   • Stop all services: Close this window or press Ctrl+C
echo.
echo TROUBLESHOOTING:
echo   • If frontend doesn't load, wait 10-15 seconds for compilation
echo   • Check the server windows for any error messages
echo   • Ensure no other applications are using ports 3000 or 8000
echo.

REM Open browser automatically
echo Opening browser...
timeout /t 2 /nobreak >nul
start http://localhost:3000

echo.
echo Press any key to stop all services and exit...
pause >nul

REM ========================================
REM CLEANUP ON EXIT
REM ========================================
echo.
echo Shutting down services...

REM Kill backend process
taskkill /FI "WindowTitle eq Trading Backend*" /T /F >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Backend server stopped
) else (
    echo [!] Backend server was not running
)

REM Kill frontend process
taskkill /FI "WindowTitle eq Trading Frontend*" /T /F >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Frontend server stopped
) else (
    echo [!] Frontend server was not running
)

REM Additional cleanup for any hanging Python/Node processes on the ports
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000"') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo All services stopped. Goodbye!
echo.
timeout /t 2 /nobreak >nul

endlocal
exit /b 0