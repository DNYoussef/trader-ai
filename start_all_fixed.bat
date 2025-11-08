@echo off
setlocal enabledelayedexpansion

cls
echo ========================================
echo   GARY-TALEB TRADING SYSTEM LAUNCHER
echo   Unified Dashboard + Trading Engine
echo ========================================
echo.

REM Check Python
where python >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)
echo [OK] Python found

REM Check Node
where node >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)
echo [OK] Node.js found

REM Check npm
where npm >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: npm is not installed or not in PATH
    pause
    exit /b 1
)
echo [OK] npm found
echo.

REM Kill any existing processes on our ports
echo Cleaning up any existing services...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3000" ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
echo [OK] Ports cleared
echo.

REM ========================================
REM BACKEND SERVER
REM ========================================
echo [1/2] Starting Backend Server...
echo.

REM Start backend in new window
start "Trading Backend" cmd /c "cd /d src\dashboard && echo Starting Backend Server... && echo. && python run_server_simple.py || (echo Backend failed to start && pause)"

REM Wait for backend
echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM ========================================
REM FRONTEND SERVER
REM ========================================
echo.
echo [2/2] Starting Frontend Server...
echo.

REM Check if node_modules exists
if not exist "src\dashboard\frontend\node_modules" (
    echo Installing frontend dependencies - this may take a few minutes...
    cd src\dashboard\frontend
    call npm install
    cd ..\..\..
    if !errorlevel! neq 0 (
        echo ERROR: Failed to install frontend dependencies
        echo Please run: cd src\dashboard\frontend ^&^& npm install
        pause
        exit /b 1
    )
    echo [OK] Frontend dependencies installed
) else (
    echo [OK] Frontend dependencies already installed
)

REM Start frontend in new window
start "Trading Frontend" cmd /c "cd /d src\dashboard\frontend && echo Starting Frontend Server... && echo. && npm run dev || (echo Frontend failed to start && pause)"

REM Wait for frontend to compile
echo Waiting for frontend to compile...
timeout /t 8 /nobreak >nul

REM ========================================
REM FINAL STATUS
REM ========================================
cls
echo ========================================
echo    SYSTEM STARTUP COMPLETE
echo ========================================
echo.
echo Services Running:
echo   - Backend API: http://localhost:8000
echo   - Frontend UI: http://localhost:3000
echo   - API Docs: http://localhost:8000/docs
echo.
echo Two new windows have been opened:
echo   - "Trading Backend" - FastAPI server
echo   - "Trading Frontend" - React development server
echo.
echo If you see any errors in those windows,
echo the servers may need troubleshooting.
echo.
echo Opening browser to http://localhost:3000...
timeout /t 2 /nobreak >nul
start http://localhost:3000

echo.
echo ========================================
echo Press any key to STOP all services...
echo ========================================
pause >nul

REM ========================================
REM CLEANUP
REM ========================================
echo.
echo Shutting down services...

REM Close the windows we opened
taskkill /FI "WindowTitle eq Trading Backend*" /T /F >nul 2>&1
taskkill /FI "WindowTitle eq Trading Frontend*" /T /F >nul 2>&1

REM Kill processes on ports as backup
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000"') do (
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3000"') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo [OK] All services stopped
echo.
pause

endlocal
exit /b 0