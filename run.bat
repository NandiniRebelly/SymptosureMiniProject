@echo off
REM Windows batch script to run the Symptoms Checker project
REM This script activates the virtual environment, runs data preparation,
REM trains the model, and starts the API server

echo ========================================
echo    Symptoms Checker - Project Runner
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo ERROR: Virtual environment not found!
    echo Please create a virtual environment first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [1/4] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment activated
echo.

echo [2/4] Running data preparation...
python data_prep.py
if errorlevel 1 (
    echo ERROR: Data preparation failed
    pause
    exit /b 1
)
echo ✓ Data preparation completed
echo.

echo [3/4] Training ML model...
python train_model.py
if errorlevel 1 (
    echo ERROR: Model training failed
    pause
    exit /b 1
)
echo ✓ Model training completed
echo.

echo [4/4] Starting API server...
echo.
echo ========================================
echo    API Server Starting...
echo ========================================
echo Server will be available at: http://localhost:8000
echo API will automatically find an available port (8000-8010)
echo Check console output for the actual port number
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the API server with automatic port detection
python api.py

REM If we get here, the server was stopped
echo.
echo API server stopped.
pause
