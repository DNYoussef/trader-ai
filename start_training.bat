@echo off
REM ========================================
REM AI TRAINING LAUNCHER - Gary×Taleb Trading System
REM Launches AI model training in a separate window
REM Prevents VSCode shutdown from interrupting training
REM ========================================

setlocal enabledelayedexpansion

echo ========================================
echo   AI MODEL TRAINING LAUNCHER
echo   Gary×Taleb Trading System
echo ========================================
echo.

REM Set colors for output (Windows 10+)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

REM Check Python installation
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%ERROR: Python is not installed or not in PATH%RESET%
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo %GREEN%✓ Python found%RESET%
python --version
echo.

REM Check for CUDA/GPU
echo %YELLOW%Checking for GPU support...%RESET%
python -c "import torch; cuda=torch.cuda.is_available(); print(f'CUDA Available: {cuda}'); print(f'GPU Count: {torch.cuda.device_count()}') if cuda else None" 2>nul
if %errorlevel% neq 0 (
    echo %YELLOW%PyTorch not installed or no GPU detected%RESET%
    echo Training will use CPU (slower)
) else (
    echo %GREEN%✓ GPU support detected%RESET%
)
echo.

REM Check for virtual environment
if exist "venv\Scripts\activate.bat" (
    echo Using virtual environment...
    call venv\Scripts\activate.bat
)

REM Create necessary directories
if not exist "models" mkdir models
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "logs" mkdir logs
if not exist "logs\training" mkdir logs\training
if not exist "trained_models" mkdir trained_models

REM Check for .env file
if not exist ".env" (
    echo %YELLOW%WARNING: .env file not found%RESET%
    echo Some models may require HF_TOKEN for Hugging Face access
    echo.
)

REM Display training options
echo %BLUE%========================================%RESET%
echo %BLUE%   SELECT TRAINING MODE%RESET%
echo %BLUE%========================================%RESET%
echo.
echo   [1] Simple ML Models (5-10 minutes)
echo   [2] Full Training Pipeline (30-60 minutes)
echo   [3] Enhanced HRM 32D Training (3-5 hours, GPU recommended)
echo   [4] Optimized HRM Training (30-60 minutes, faster)
echo   [5] Resume Previous Training
echo   [6] Quick Data Test (1-2 minutes)
echo   [7] Monitor Existing Training
echo   [8] Custom Script
echo   [Q] Quit
echo.

choice /C 12345678Q /N /M "Select option: "
set TRAINING_CHOICE=%errorlevel%

if %TRAINING_CHOICE%==9 goto :END

REM Set script based on choice
if %TRAINING_CHOICE%==1 (
    set SCRIPT=scripts\training\simple_train.py
    set DESCRIPTION=Simple ML Models Training
    set ESTIMATED_TIME=5-10 minutes
)
if %TRAINING_CHOICE%==2 (
    set SCRIPT=scripts\training\execute_training.py
    set DESCRIPTION=Full Training Pipeline
    set ESTIMATED_TIME=30-60 minutes
)
if %TRAINING_CHOICE%==3 (
    set SCRIPT=scripts\training\train_enhanced_hrm_32d.py
    set DESCRIPTION=Enhanced HRM 32D Training
    set ESTIMATED_TIME=3-5 hours
)
if %TRAINING_CHOICE%==4 (
    set SCRIPT=scripts\training\train_optimized_hrm_32d.py
    set DESCRIPTION=Optimized HRM Training
    set ESTIMATED_TIME=30-60 minutes
)
if %TRAINING_CHOICE%==5 (
    set SCRIPT=scripts\training\train_enhanced_hrm_32d_resume.py
    set DESCRIPTION=Resume Previous Training
    set ESTIMATED_TIME=Varies
)
if %TRAINING_CHOICE%==6 (
    set SCRIPT=scripts\training\quick_data_test.py
    set DESCRIPTION=Quick Data Test
    set ESTIMATED_TIME=1-2 minutes
)
if %TRAINING_CHOICE%==7 (
    set SCRIPT=scripts\training\monitor_live_training.py
    set DESCRIPTION=Training Monitor
    set ESTIMATED_TIME=Continuous
)
if %TRAINING_CHOICE%==8 (
    set /P SCRIPT="Enter script path: "
    set DESCRIPTION=Custom Training Script
    set ESTIMATED_TIME=Unknown
)

REM Confirm selection
echo.
echo %YELLOW%========================================%RESET%
echo %YELLOW%   TRAINING CONFIGURATION%RESET%
echo %YELLOW%========================================%RESET%
echo.
echo   Script: %SCRIPT%
echo   Description: %DESCRIPTION%
echo   Estimated Time: %ESTIMATED_TIME%
echo.

choice /M "Start training"
if %errorlevel% neq 1 goto :END

REM Check if script exists
if not exist "%SCRIPT%" (
    echo %RED%ERROR: Script not found: %SCRIPT%%RESET%
    pause
    goto :END
)

REM Install dependencies if needed
echo.
echo %YELLOW%Checking dependencies...%RESET%
pip show torch >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install -r requirements.txt
    if exist "src\intelligence\requirements.txt" (
        pip install -r src\intelligence\requirements.txt
    )
)

REM Launch training in new window
echo.
echo %GREEN%========================================%RESET%
echo %GREEN%   LAUNCHING TRAINING%RESET%
echo %GREEN%========================================%RESET%
echo.
echo Training will run in a new window.
echo You can safely close VSCode or this window.
echo Training will continue in the background.
echo.

REM Create timestamp for log file
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%

REM Start training in new window with logging
start "AI Training - %DESCRIPTION%" cmd /k "echo Starting %DESCRIPTION% at %TIME% && echo. && python %SCRIPT% 2>&1 | tee logs\training\training_%TIMESTAMP%.log && echo. && echo Training completed at %TIME% && echo Press any key to close this window... && pause >nul"

REM Show monitoring instructions
echo %YELLOW%========================================%RESET%
echo %YELLOW%   TRAINING STARTED%RESET%
echo %YELLOW%========================================%RESET%
echo.
echo Training is now running in a separate window.
echo.
echo %BLUE%TO MONITOR:%RESET%
echo   • Check the training window for real-time output
echo   • View logs: logs\training\training_%TIMESTAMP%.log
echo   • Run monitoring script: python scripts\training\monitor_live_training.py
echo.
echo %BLUE%OUTPUT LOCATIONS:%RESET%
echo   • Models: models\ and trained_models\
echo   • Checkpoints: models\checkpoints\
echo   • Metrics: models\training_metrics_live.json
echo.
echo %YELLOW%IMPORTANT:%RESET%
echo   • Do NOT close the training window unless you want to stop training
echo   • You can close this launcher window safely
echo   • Training will continue even if VSCode is closed
echo.

REM Option to open monitoring
choice /T 10 /D N /M "Open training monitor in another window"
if %errorlevel%==1 (
    start "Training Monitor" cmd /k "python scripts\training\monitor_live_training.py"
)

:END
echo.
echo %GREEN%Launcher closing. Training continues in background if started.%RESET%
timeout /t 3 /nobreak >nul

endlocal
exit /b 0