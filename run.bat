@echo off
REM Creates a virtual environment on first use, installs the pinned packages and runs the pipeline.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    python -m venv .venv || (echo Python 3.10-3.13 is required. & pause & exit /b 1)
)
.venv\Scripts\python -m pip install --disable-pip-version-check -q -r requirements.txt || (pause & exit /b 1)
.venv\Scripts\python run_pipeline.py %*
pause
