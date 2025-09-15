@echo off
REM Quick startup script for Gary×Taleb Trading Dashboard UI
REM This starts both backend and frontend servers

echo 🚀 Starting Gary×Taleb Trading Dashboard...
echo.

REM Start backend server
echo 📡 Starting Backend Server (Port 8000)...
start cmd /k "cd src\dashboard && python run_server_simple.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend development server
echo 🎨 Starting Frontend Server (Port 3000)...
start cmd /k "cd src\dashboard\frontend && npm run dev"

echo.
echo ✅ Dashboard servers starting...
echo.
echo 📊 Frontend will be available at: http://localhost:3000
echo 📡 Backend API available at: http://localhost:8000
echo 🔌 WebSocket endpoint: ws://localhost:8000/ws/{client_id}
echo.
echo 💡 Tips:
echo   - Frontend may take a moment to compile on first load
echo   - Check browser console for any connection issues
echo   - Both servers will run in separate command windows
echo.
echo Press any key to open the dashboard in your browser...
pause >nul

start http://localhost:3000

echo.
echo Dashboard opened in browser!
echo Close the command windows to stop the servers.