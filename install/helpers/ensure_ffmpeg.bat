@echo off
setlocal enabledelayedexpansion
REM Ensure ffmpeg is available (required for all audio/video processing).
REM Called by both auto_install.bat and update.bat so a single command always
REM leaves the environment complete. Non-fatal: prints an ACTION REQUIRED note
REM if ffmpeg is still missing and lets the caller continue.
REM
REM Note: a freshly winget-installed ffmpeg lands on the persistent user PATH,
REM so a NEW terminal (and the GUI launched from one) finds it. ffmpeg is only
REM needed at conversion time, not during install, so not propagating PATH back
REM to the caller's current session is fine.

where ffmpeg >nul 2>&1
if !errorlevel! equ 0 exit /b 0

set "DO_FFMPEG_INSTALL=1"
where winget >nul 2>&1
if !errorlevel! neq 0 (
    echo winget is not available on this system.
    set "DO_FFMPEG_INSTALL="
)
if defined DO_FFMPEG_INSTALL (
    REM License transparency: the winget flags below accept the package's
    REM license terms on the user's behalf - announce it and, when the
    REM session is interactive, ask first.
    echo ffmpeg not found. It can be installed automatically via winget
    echo ^(package 'Gyan.FFmpeg', a GPL build^). Proceeding accepts that
    echo package's license terms on your behalf.
    powershell -NoProfile -Command "if ([Console]::IsInputRedirected) { exit 1 } else { exit 0 }" >nul 2>&1
    if !errorlevel! equ 0 (
        set "REPLY="
        set /p "REPLY=Install ffmpeg via winget now? [Y/n]: "
        if /i "!REPLY:~0,1!"=="n" set "DO_FFMPEG_INSTALL="
    ) else (
        echo Non-interactive session - installing automatically.
    )
)
if defined DO_FFMPEG_INSTALL (
    winget install --id Gyan.FFmpeg -e --silent --accept-package-agreements --accept-source-agreements
    REM Make freshly installed winget shims available in THIS session
    set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;!PATH!"
)
where ffmpeg >nul 2>&1
if !errorlevel! neq 0 (
    echo.
    echo ------------------------------------------------------------
    echo  ACTION REQUIRED - ffmpeg is required for all audio/video
    echo  processing. UltraSinger will NOT work until this is done:
    echo    1. Download ffmpeg from https://www.ffmpeg.org/download.html
    echo       ^(Windows builds: gyan.dev or BtbN^)
    echo    2. Put ffmpeg.exe on your PATH ^(or re-run in a NEW terminal
    echo       after installing via winget^)
    echo ------------------------------------------------------------
    echo.
) else (
    echo ffmpeg is now available.
)
exit /b 0
