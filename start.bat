@echo off
rem English Echo Trainer launcher (pure ASCII to avoid codepage issues)
set "CONDA_DIR=%USERPROFILE%\miniconda3"
if not exist "%CONDA_DIR%" set "CONDA_DIR=%USERPROFILE%\anaconda3"
if not exist "%CONDA_DIR%\envs\english\python.exe" (
    echo [ERROR] Conda env "english" not found under %CONDA_DIR%
    echo         Run setup first - see README.md section "Installation".
    pause
    exit /b 1
)
set "ENV=%CONDA_DIR%\envs\english"
set "PATH=%ENV%;%ENV%\Library\bin;%ENV%\Scripts;%PATH%"
cd /d "%~dp0"
echo Starting English Echo Trainer ...
echo Close this window to stop the app.
"%ENV%\python.exe" -m streamlit run app.py
pause
