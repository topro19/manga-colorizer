@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo          MangaColorizer - Automated Windows Setup for AMD GPU
echo ======================================================================
echo.

:: 1. Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in PATH!
    echo Please install Python 3.12 (64-bit) from python.org or the Microsoft Store.
    pause
    exit /b 1
)

echo [1/7] Python detected:
python --version
echo.

:: 2. Create virtual environment
if not exist "venv" (
    echo [2/7] Creating virtual environment (venv)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [2/7] Virtual environment already exists.
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: 3. Install dependencies
echo [3/7] Installing dependencies from requirements.txt...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo.

:: 4. Download ONNX model if missing
if not exist "models\manga-colorize-fp16.onnx" (
    echo [4/7] Downloading MangaColorization-v2 FP16 ONNX model (~58.8 MB)...
    if not exist "models" mkdir models
    curl.exe -L "https://huggingface.co/Faridzar/manga-colorization-v2-onnx/resolve/main/manga-colorize-fp16.onnx" -o "models\manga-colorize-fp16.onnx"
    if %errorlevel% neq 0 (
        echo [ERROR] Model download failed. Please check your internet connection.
        pause
        exit /b 1
    )
) else (
    echo [4/7] Model already present: models\manga-colorize-fp16.onnx
)
echo.

:: 5. Test ONNX Runtime
echo [5/7] Testing ONNX Runtime providers...
python -c "import onnxruntime as ort; print('Available ONNX Execution Providers:', ort.get_available_providers())"
echo.

:: 6. Test DirectML on AMD GPU
echo [6/7] Testing DirectML on AMD Radeon GPU...
python -c "from core.model_manager import ModelManager; m = ModelManager(); res = m.test_gpu(); print(res['message'])"
echo.

:: 7. Launch application
echo [7/7] Launching MangaColorizer Desktop Application...
echo.
python app.py

pause
