@echo off
echo ========================================
echo   STARTING TRADING SYSTEM
echo ========================================
echo.

echo [1] Starting Backend Server...
start cmd /k "cd /d src\dashboard && python run_server_simple.py"

echo [2] Waiting 5 seconds for backend...
timeout /t 5

echo [3] Starting Frontend Server...
start cmd /k "cd /d src\dashboard\frontend && npm run dev"

echo.
echo ========================================
echo SERVERS ARE STARTING IN NEW WINDOWS
echo ========================================
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Opening browser in 10 seconds...
timeout /t 10

start http://localhost:3000

echo.
echo All services started!
echo Close this window when done.
pause