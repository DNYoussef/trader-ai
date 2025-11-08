@echo off
REM ========================================
REM STOP ALL SERVICES - Gary×Taleb Trading System
REM Kills all backend and frontend processes
REM ========================================

echo ========================================
echo   STOPPING ALL TRADING SYSTEM SERVICES
echo ========================================
echo.

echo Stopping services on ports 8000 and 3000...

REM Kill any Python processes on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Stopping process %%a on port 8000...
    taskkill /PID %%a /F >nul 2>&1
    if !errorlevel! == 0 (
        echo [OK] Process %%a stopped
    )
)

REM Kill any Node processes on port 3000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000" ^| findstr "LISTENING"') do (
    echo Stopping process %%a on port 3000...
    taskkill /PID %%a /F >nul 2>&1
    if !errorlevel! == 0 (
        echo [OK] Process %%a stopped
    )
)

REM Kill any windows with specific titles
taskkill /FI "WindowTitle eq Trading Backend*" /T /F >nul 2>&1
taskkill /FI "WindowTitle eq Trading Frontend*" /T /F >nul 2>&1
taskkill /FI "WindowTitle eq Enhanced Trading UI*" /T /F >nul 2>&1

REM Additional cleanup for python and node processes
taskkill /IM python.exe /F >nul 2>&1
taskkill /IM node.exe /F >nul 2>&1

echo.
echo All services stopped.
echo.
echo You can now run start_all.bat to restart the system.
echo.
pause