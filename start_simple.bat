@echo off
REM Simple startup script with debugging

echo ========================================
echo SIMPLE STARTUP - TRADING SYSTEM
echo ========================================
echo.

REM Start backend
echo Starting backend server...
echo Command: cd src\dashboard ^&^& python run_server_simple.py
echo.
start "Backend Server" cmd /k "cd /d %CD%\src\dashboard && python run_server_simple.py"

REM Wait a bit
timeout /t 5 /nobreak

REM Start frontend
echo Starting frontend server...
echo Command: cd src\dashboard\frontend ^&^& npm run dev
echo.
start "Frontend Server" cmd /k "cd /d %CD%\src\dashboard\frontend && npm run dev"

echo.
echo ========================================
echo SERVERS STARTING
echo ========================================
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Check the opened windows for any errors.
echo Press any key to open the dashboard...
pause >nul

start http://localhost:3000

echo.
echo System is running. Press any key to stop all services...
pause >nul

REM Cleanup
taskkill /FI "WindowTitle eq Backend Server*" /T /F
taskkill /FI "WindowTitle eq Frontend Server*" /T /F

echo Done!
pause