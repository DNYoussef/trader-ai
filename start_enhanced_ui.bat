@echo off
echo ========================================
echo Starting Enhanced Trading Dashboard
echo With Mobile Psychology Integration
echo ========================================

:: Set working directory
cd /d "%~dp0"

:: Start the backend server
echo.
echo Starting backend server...
start "Trading Backend" cmd /k "cd src\dashboard && python run_server_simple.py"

:: Wait for backend to initialize
timeout /t 3 /nobreak > nul

:: Install frontend dependencies if needed
echo.
echo Checking frontend dependencies...
cd src\dashboard\frontend
if not exist node_modules (
    echo Installing dependencies...
    call npm install
)

:: Update main.tsx to use enhanced version
echo.
echo Configuring enhanced UI...
copy /Y src\main-enhanced.tsx src\main.tsx > nul 2>&1

:: Start the frontend with enhanced UI
echo.
echo Starting enhanced frontend...
start "Enhanced Trading UI" cmd /k "npm run dev"

:: Wait for frontend to start
timeout /t 3 /nobreak > nul

:: Open browser
echo.
echo Opening browser...
start http://localhost:3000

echo.
echo ========================================
echo Enhanced Trading Dashboard is running!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Press any key to stop all services...
pause > nul

:: Restore original main.tsx
copy /Y src\main.tsx.backup src\main.tsx > nul 2>&1

:: Kill both processes
taskkill /FI "WindowTitle eq Trading Backend*" /T /F > nul 2>&1
taskkill /FI "WindowTitle eq Enhanced Trading UI*" /T /F > nul 2>&1

echo.
echo All services stopped.
pause