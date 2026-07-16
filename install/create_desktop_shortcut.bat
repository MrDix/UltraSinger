@echo off
REM Creates Desktop and Start Menu shortcuts for the UltraSinger GUI
REM (UltraSinger icon, taskbar-pinnable). Safe to run any time; existing
REM shortcuts are simply overwritten. Called by the installer after an
REM interactive confirmation, or run it directly by double-clicking.
REM Arg %1 = "nopause" suppresses the final pause (installer mode).

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0helpers\create_gui_shortcut.ps1"
set "RC=%errorlevel%"
if /i not "%~1"=="nopause" pause
exit /b %RC%
