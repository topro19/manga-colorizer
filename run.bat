@echo off
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)
python app.py
if %errorlevel% neq 0 (
    pause
)
