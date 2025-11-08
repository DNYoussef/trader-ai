@echo off
echo ========================================
echo Testing Real Data Integration
echo ========================================
echo.

echo [1] Testing Feature Calculator...
python -c "from src.dashboard.feature_calculator import Feature32Calculator; calc = Feature32Calculator(); features = calc.get_all_features(); print(f'Features: {len(features.get(\"values\", []))} dimensions'); print(f'First 5 values: {features.get(\"values\", [])[:5]}')"
echo.

echo [2] Testing AI Model Service...
python -c "from src.dashboard.ai_model_service import get_model_service; service = get_model_service(); info = service.get_model_info(); print(f'Model loaded: {info.get(\"model_loaded\")}'); print(f'Available strategies: {info.get(\"strategies\", [])[:3]}...')"
echo.

echo [3] Testing Backend API Endpoints...
echo Starting backend server in background...
start /B cmd /c "cd src\dashboard && python run_server_simple.py > nul 2>&1"
timeout /t 3 > nul

echo Testing /api/features/realtime endpoint...
curl -s http://localhost:8000/api/features/realtime | python -m json.tool | findstr "values timestamp"
echo.

echo Testing /api/ai/model/info endpoint...
curl -s http://localhost:8000/api/ai/model/info | python -m json.tool | findstr "model_loaded strategies"
echo.

echo [4] Frontend Component Check...
cd src\dashboard\frontend
echo Checking for required components...
if exist "src\components\Feature32Panel.tsx" (
    echo [OK] Feature32Panel.tsx found
) else (
    echo [ERROR] Feature32Panel.tsx missing
)

if exist "src\components\AIStrategyPanel.tsx" (
    echo [OK] AIStrategyPanel.tsx found
) else (
    echo [ERROR] AIStrategyPanel.tsx missing
)

if exist "src\components\TradingControls.tsx" (
    echo [OK] TradingControls.tsx found
) else (
    echo [ERROR] TradingControls.tsx missing
)

echo.
echo [5] Installing Three.js dependencies...
echo Running npm install for new packages...
call npm install three @react-three/fiber @react-three/drei --save
echo.

echo ========================================
echo Test Complete!
echo ========================================
echo.
echo To start the full system with real data:
echo   1. Run start_all.bat
echo   2. Open http://localhost:3000
echo   3. Click on "Trading Terminal" tab
echo.
echo Features:
echo   - Real 32 AI input features displayed in 3D
echo   - AI strategy recommendations based on trained model
echo   - Functional trading controls with dropdowns and timeframes
echo   - Real market data from SQLite databases
echo.
pause