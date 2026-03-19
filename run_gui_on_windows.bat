@echo off
:: Launch UltraSinger GUI on Windows
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found at .venv
    echo Please run one of the installation scripts first:
    echo   - install\CPU\windows_cpu.bat ^(for CPU^)
    echo   - install\CUDA\windows_cuda_gpu.bat ^(for GPU with CUDA^)
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
cd src
echo Starting UltraSinger GUI...
python gui_main.py
