@echo off
:: Launch UltraSinger CLI on Windows
cd /d "%~dp0"

:: Check if .venv exists (install script must be run first)
if not exist .venv (
    echo ============================================================
    echo  ERROR: Virtual environment not found.
    echo ============================================================
    echo.
    echo  You need to run the installation script first:
    echo.
    echo    install\CPU\windows_cpu.bat       ^(CPU-only^)
    echo    install\CUDA\windows_cuda_gpu.bat ^(NVIDIA GPU with CUDA^)
    echo.
    echo  Prerequisites:
    echo    - Python 3.12 or 3.13  https://www.python.org/downloads/
    echo    - ffmpeg in PATH       https://www.ffmpeg.org/download.html
    echo.
    echo  See README.md section "How to use this source code" for details.
    echo ============================================================
    pause
    exit /b 1
)

:: Check if ffmpeg is available
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo ============================================================
    echo  WARNING: ffmpeg not found in PATH.
    echo ============================================================
    echo.
    echo  UltraSinger requires ffmpeg for audio/video processing.
    echo  Download: https://www.ffmpeg.org/download.html
    echo  Make sure ffmpeg.exe is in your system PATH.
    echo.
    echo  You can also specify the path with: --ffmpeg "C:\path\to\ffmpeg"
    echo ============================================================
    echo.
)

:: Activate the virtual environment and open cmd in the src directory
:: Suppress compile-time SyntaxWarnings from third-party packages (e.g. pydub)
set PYTHONWARNINGS=ignore::SyntaxWarning
call .venv\Scripts\activate.bat
cd src
echo Starting UltraSinger...
echo Use: python UltraSinger.py -h  for help
echo.
cmd /k
