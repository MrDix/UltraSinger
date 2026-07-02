@echo off
REM Set up the bgutil PO-token provider (Node.js server) so YouTube downloads
REM get full-quality formats. Non-fatal: if Node.js is missing the rest of the
REM install still succeeds and the GUI shows a setup hint instead.
REM Must be run from the repository root (the install scripts cd there first).

setlocal enabledelayedexpansion
set "PROVIDER_DIR=.potoken\bgutil-ytdlp-pot-provider"
set "PROVIDER_REPO=https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git"
set "SERVER_ENTRY=%PROVIDER_DIR%\server\build\main.js"

echo.
echo Setting up the PO-token provider (full-quality YouTube downloads)...

where node >nul 2>&1
if !errorlevel! neq 0 (
    echo   Node.js not found - skipping. Install Node.js from https://nodejs.org
    echo   and re-run this script to enable full-quality YouTube downloads.
    exit /b 0
)
where git >nul 2>&1
if !errorlevel! neq 0 (
    echo   git not found - skipping PO-token provider setup.
    exit /b 0
)

if not exist ".potoken" mkdir ".potoken"
if exist "%PROVIDER_DIR%\.git" (
    echo   Updating provider source...
    git -C "%PROVIDER_DIR%" pull --ff-only >nul 2>&1
) else (
    echo   Cloning provider source...
    git clone --depth 1 "%PROVIDER_REPO%" "%PROVIDER_DIR%" >nul 2>&1
)
if not exist "%PROVIDER_DIR%\server" (
    echo   Could not obtain the provider source - skipping.
    exit /b 0
)

echo   Building provider (npm install + tsc)...
pushd "%PROVIDER_DIR%\server"
call npm install --no-audit --no-fund >nul 2>&1
call npx --yes tsc >nul 2>&1
popd

if exist "%SERVER_ENTRY%" (
    echo   PO-token provider ready. The GUI starts it automatically on launch.
) else (
    echo   PO-token provider build did not complete - the GUI will show a hint.
)
exit /b 0
