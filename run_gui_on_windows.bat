@echo off
:: Launch UltraSinger GUI on Windows
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo Error: uv is not installed or not in PATH.
    echo Install uv: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

set UV_LINK_MODE=copy
echo Starting UltraSinger GUI...
uv run --extra gui python src/gui_main.py
