@echo off
REM Set up the bgutil PO-token provider (Node.js server) so YouTube downloads
REM get full-quality formats. Non-fatal: the parent install script keeps going
REM regardless. Exit codes: 0 = provider ready, 2 = Node.js/git missing,
REM 3 = build failed. Arg %1 = the install script to name in the re-run hint.
REM Must be run from the repository root (the install scripts cd there first).

setlocal enabledelayedexpansion
set "RERUN=%~1"
if "%RERUN%"=="" set "RERUN=the install script you just ran"
set "PROVIDER_DIR=.potoken\bgutil-ytdlp-pot-provider"
set "PROVIDER_REPO=https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git"
set "SERVER_ENTRY=%PROVIDER_DIR%\server\build\main.js"

echo.
echo ============================================================
echo  PO-token provider setup (full-quality YouTube downloads)
echo ============================================================

REM --- Ensure Node.js is available (auto-install via winget if missing) ---
where node >nul 2>&1
if !errorlevel! neq 0 (
    echo Node.js not found. Trying to install it automatically via winget...
    where winget >nul 2>&1
    if !errorlevel! equ 0 (
        winget install --id OpenJS.NodeJS.LTS -e --silent --accept-package-agreements --accept-source-agreements
        REM Make the freshly installed node available in THIS session
        set "PATH=%ProgramFiles%\nodejs;%PATH%"
    ) else (
        echo winget is not available on this system.
    )
    where node >nul 2>&1
    if !errorlevel! neq 0 (
        echo.
        echo ------------------------------------------------------------
        echo  ACTION REQUIRED - Node.js is needed for full-quality
        echo  YouTube downloads. UltraSinger still works, but YouTube
        echo  downloads stay limited to 360p until this is done:
        echo.
        echo    1. Install Node.js LTS from https://nodejs.org
        echo       ^(choose the "LTS" installer, keep all defaults^)
        echo    2. Close this window and open a NEW terminal
        echo       ^(so Windows picks up Node.js on the PATH^)
        echo    3. Run the installer again:
        echo         %RERUN%
        echo ------------------------------------------------------------
        echo.
        exit /b 2
    )
    echo Node.js is now available.
)

where git >nul 2>&1
if !errorlevel! neq 0 (
    echo.
    echo ------------------------------------------------------------
    echo  git was not found, so the provider source cannot be fetched.
    echo  Install Git from https://git-scm.com, open a NEW terminal,
    echo  and run the installer again:
    echo    %RERUN%
    echo ------------------------------------------------------------
    echo.
    exit /b 2
)

REM --- Fetch and build the provider ---
if not exist ".potoken" mkdir ".potoken"
if exist "%PROVIDER_DIR%\.git" (
    echo Updating provider source...
    git -C "%PROVIDER_DIR%" pull --ff-only >nul 2>&1
) else (
    echo Downloading provider source...
    git clone --depth 1 "%PROVIDER_REPO%" "%PROVIDER_DIR%" >nul 2>&1
)
if not exist "%PROVIDER_DIR%\server" (
    echo Could not download the provider source. Check your internet connection
    echo and run %RERUN% again.
    echo If you are behind a corporate proxy: git and npm honor the same
    echo HTTP_PROXY/HTTPS_PROXY/NO_PROXY variables as the rest of this installer
    echo - set them ^(and UV_SYSTEM_CERTS=1 for TLS-inspecting proxies^) and re-run.
    exit /b 3
)

echo Building provider ^(npm install + tsc, this can take a minute^)...
pushd "%PROVIDER_DIR%\server"
call npm install --no-audit --no-fund >nul 2>&1
call npx --yes tsc >nul 2>&1
popd

if exist "%SERVER_ENTRY%" (
    echo Done. The GUI starts the provider automatically on launch -
    echo full-quality YouTube downloads are enabled.
    exit /b 0
) else (
    echo The provider build did not complete. Run %RERUN% again;
    echo if it keeps failing, YouTube downloads stay limited to 360p.
    echo If you are behind a corporate proxy: git and npm honor the same
    echo HTTP_PROXY/HTTPS_PROXY/NO_PROXY variables as the rest of this installer
    echo - set them ^(and UV_SYSTEM_CERTS=1 for TLS-inspecting proxies^) and re-run.
    exit /b 3
)
