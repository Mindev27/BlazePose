@echo off
echo Activating BlazePose virtual environment...

REM Check if the virtual environment exists
if not exist "blazepose_env\Scripts\activate.bat" (
    echo Creating new virtual environment...
    python -m venv blazepose_env
)

REM Activate the virtual environment
call blazepose_env\Scripts\activate.bat

REM Check if requirements are installed
echo Checking dependencies...
pip freeze > installed.txt
findstr /c:"numpy" /c:"opencv-python" /c:"mediapipe" /c:"matplotlib" /c:"PyQt5" installed.txt > nul
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)
del installed.txt

REM Run the application launcher
echo Starting BlazePose application...
python run.py

REM Deactivate the virtual environment when done
call blazepose_env\Scripts\deactivate.bat

echo Done.
pause 