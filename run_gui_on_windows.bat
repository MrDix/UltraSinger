@echo off
:: Launch UltraSinger GUI on Windows
cd /d "%~dp0"

:: Check if uv is available
where uv >nul 2>&1
if errorlevel 1 (
    echo ============================================================
    echo  ERROR: uv is not installed or not in PATH.
    echo ============================================================
    echo.
    echo  The GUI requires uv ^(Python package manager^).
    echo.
    echo  You need to run the installation script first:
    echo.
    echo    install\CPU\windows_cpu.bat       ^(CPU-only^)
    echo    install\CUDA\windows_cuda_gpu.bat ^(NVIDIA GPU with CUDA^)
    echo.
    echo  The install script will install uv automatically.
    echo  Or install manually: https://docs.astral.sh/uv/getting-started/installation/
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
    echo ============================================================
    echo.
)

set UV_LINK_MODE=copy
:: Suppress compile-time SyntaxWarnings from third-party packages (e.g. pydub)
set PYTHONWARNINGS=ignore::SyntaxWarning
echo Starting UltraSinger GUI...
uv run --extra gui python src/gui_main.py
