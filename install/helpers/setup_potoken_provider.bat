@echo off
REM Set up the bgutil PO-token provider (Node.js server) so video downloads
REM get full-quality formats. Non-fatal: the parent install script keeps going
REM regardless. Exit codes: 0 = provider ready, 2 = Node.js/git missing or
REM Node.js too old, 3 = download or build failed. Arg %1 = the install script
REM to name in the re-run hint.
REM Must be run from the repository root (the install scripts cd there first).

setlocal enabledelayedexpansion
set "RERUN=%~1"
if "%RERUN%"=="" set "RERUN=the install script you just ran"
set "PROVIDER_DIR=.potoken\bgutil-ytdlp-pot-provider"
set "PROVIDER_REPO=https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git"
REM The provider release that is built, pinned by its tag and verified by its
REM commit, so neither a newer release nor a moved tag runs unreviewed code.
REM Keep it at the version of the bgutil-ytdlp-pot-provider plugin in uv.lock
REM (server and plugin are released together) and in sync with
REM setup_potoken_provider.sh. NODE_MIN_MAJOR is the release's "engines" entry.
set "PROVIDER_TAG=2.0.2"
set "PROVIDER_COMMIT=26475e9d3665b972c4aeb83b4dbaafcf64b88438"
set "NODE_MIN_MAJOR=22"
set "SERVER_DIR=%PROVIDER_DIR%\server"
set "SERVER_ENTRY=%SERVER_DIR%\build\main.js"
REM The commit the current build was made from
set "BUILD_STAMP=%SERVER_DIR%\build\.ultrasinger-commit"

echo.
echo ============================================================
echo  PO-token provider setup (full-quality video downloads)
echo ============================================================

REM --- Ensure Node.js is available (auto-install via winget if missing) ---
where node >nul 2>&1
if !errorlevel! neq 0 (
    set "DO_NODE_INSTALL=1"
    where winget >nul 2>&1
    if !errorlevel! neq 0 (
        echo winget is not available on this system.
        set "DO_NODE_INSTALL="
    )
    if defined DO_NODE_INSTALL (
        REM License transparency: the winget flags below accept the package's
        REM license terms on the user's behalf - announce it and, when the
        REM session is interactive, ask first.
        echo Node.js not found. It can be installed automatically via winget
        echo ^(package 'OpenJS.NodeJS.LTS', MIT-licensed^). Proceeding accepts
        echo that package's license terms on your behalf.
        powershell -NoProfile -Command "if ([Console]::IsInputRedirected) { exit 1 } else { exit 0 }" >nul 2>&1
        if !errorlevel! equ 0 (
            set "REPLY="
            set /p "REPLY=Install Node.js via winget now? [Y/n]: "
            if /i "!REPLY:~0,1!"=="n" set "DO_NODE_INSTALL="
        ) else (
            echo Non-interactive session - installing automatically.
        )
    )
    if defined DO_NODE_INSTALL (
        winget install --id OpenJS.NodeJS.LTS -e --silent --accept-package-agreements --accept-source-agreements
        REM Make the freshly installed node available in THIS session. The
        REM installer updates the registry PATH, but this console keeps its
        REM old copy - probe the known install locations (machine-wide and
        REM per-user) and refresh PATH from the registry as a fallback.
        REM Without this, the check below fails right after a successful
        REM install and the provider stays unbuilt until the next run.
        if exist "%ProgramFiles%\nodejs\node.exe" set "PATH=%ProgramFiles%\nodejs;!PATH!"
        if exist "%LOCALAPPDATA%\Programs\nodejs\node.exe" set "PATH=%LOCALAPPDATA%\Programs\nodejs;!PATH!"
        where node >nul 2>&1
        if !errorlevel! neq 0 (
            for /f "delims=" %%P in ('powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')"') do set "PATH=%%P;!PATH!"
        )
    )
    where node >nul 2>&1
    if !errorlevel! neq 0 (
        echo.
        echo ------------------------------------------------------------
        echo  ACTION REQUIRED - Node.js is needed for full-quality
        echo  video downloads. UltraSinger still works, but video
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

REM The pinned provider release needs a recent Node.js
set "NODE_MAJOR="
for /f "delims=" %%M in ('node -p "parseInt(process.versions.node)" 2^>nul') do set "NODE_MAJOR=%%M"
set /a "NODE_MAJOR=NODE_MAJOR+0" >nul 2>&1
if !NODE_MAJOR! LSS %NODE_MIN_MAJOR% (
    set "NODE_FOUND="
    for /f "delims=" %%V in ('node --version 2^>nul') do set "NODE_FOUND=%%V"
    echo.
    echo ------------------------------------------------------------
    echo  ACTION REQUIRED - Node.js %NODE_MIN_MAJOR% or newer is needed for
    echo  full-quality video downloads ^(found !NODE_FOUND!^).
    echo  The provider is not updated until this is done:
    echo.
    echo    1. Install the current Node.js LTS from https://nodejs.org
    echo    2. Close this window and open a NEW terminal
    echo    3. Run the installer again:
    echo         %RERUN%
    echo ------------------------------------------------------------
    echo.
    exit /b 2
)

REM Leftovers of an interrupted run: the download, the replaced folder, and the
REM new or replaced build
if exist "%PROVIDER_DIR%.new" rmdir /s /q "%PROVIDER_DIR%.new"
if exist "%PROVIDER_DIR%.old" rmdir /s /q "%PROVIDER_DIR%.old"
if exist "%SERVER_DIR%\build.new" rmdir /s /q "%SERVER_DIR%\build.new"
if exist "%SERVER_DIR%\build.old" rmdir /s /q "%SERVER_DIR%\build.old"

REM --- Already built from the pinned release? ---
REM Also needs the installed dependencies (npm ci writes the hidden lockfile):
REM security software sometimes removes them, and then the build must be redone.
if exist "%SERVER_ENTRY%" if exist "%SERVER_DIR%\node_modules\.package-lock.json" if exist "%BUILD_STAMP%" (
    set "BUILT_COMMIT="
    set /p BUILT_COMMIT=<"%BUILD_STAMP%"
    if "!BUILT_COMMIT!"=="%PROVIDER_COMMIT%" (
        echo The provider ^(release %PROVIDER_TAG%^) is already built -
        echo full-quality video downloads are enabled.
        exit /b 0
    )
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

REM --- Fetch the pinned release ---
REM Every git command on the provider folder checks for its .git first: in a
REM folder without one, git would act on the UltraSinger repository instead.
if not exist ".potoken" mkdir ".potoken"
set "FOLDER_IN_USE="
if exist "%PROVIDER_DIR%\.git" (
    echo Updating the provider source to release %PROVIDER_TAG%...
    git -C "%PROVIDER_DIR%" fetch --quiet --depth 1 --force origin "refs/tags/%PROVIDER_TAG%:refs/tags/%PROVIDER_TAG%" >nul 2>&1
) else (
    echo Downloading the provider source ^(release %PROVIDER_TAG%^)...
    REM Into a new folder first: a provider folder without .git (e.g. restored
    REM from a backup that skips .git) is only replaced once the download worked,
    REM and by renaming it, which fails as a whole while the folder is in use
    REM where deleting it would leave a damaged folder behind.
    git clone --quiet --depth 1 --branch "%PROVIDER_TAG%" "%PROVIDER_REPO%" "%PROVIDER_DIR%.new" >nul 2>&1
    set "NEW_COMMIT="
    if exist "%PROVIDER_DIR%.new\.git" (
        for /f "delims=" %%H in ('git -C "%PROVIDER_DIR%.new" rev-parse HEAD 2^>nul') do set "NEW_COMMIT=%%H"
    )
    if "!NEW_COMMIT!"=="%PROVIDER_COMMIT%" (
        REM move only into a free name: it would put the folder inside a leftover one
        if exist "%PROVIDER_DIR%" if not exist "%PROVIDER_DIR%.old" move "%PROVIDER_DIR%" "%PROVIDER_DIR%.old" >nul 2>&1
        if exist "%PROVIDER_DIR%" (
            set "FOLDER_IN_USE=1"
        ) else (
            move "%PROVIDER_DIR%.new" "%PROVIDER_DIR%" >nul 2>&1
            if exist "%PROVIDER_DIR%" (
                if exist "%PROVIDER_DIR%.old" rmdir /s /q "%PROVIDER_DIR%.old"
            ) else (
                if exist "%PROVIDER_DIR%.old" move "%PROVIDER_DIR%.old" "%PROVIDER_DIR%" >nul 2>&1
            )
        )
    )
    if exist "%PROVIDER_DIR%.new" rmdir /s /q "%PROVIDER_DIR%.new"
)
if defined FOLDER_IN_USE (
    echo The provider folder %PROVIDER_DIR% is in use ^(close UltraSinger,
    echo or wait for the virus scanner to finish^) - run %RERUN% again.
    exit /b 3
)
set "SRC_COMMIT="
if exist "%PROVIDER_DIR%\.git" (
    REM Check out exactly the pinned release; this also drops local changes to it
    git -C "%PROVIDER_DIR%" -c advice.detachedHead=false checkout --quiet --force "refs/tags/%PROVIDER_TAG%" >nul 2>&1
    for /f "delims=" %%H in ('git -C "%PROVIDER_DIR%" rev-parse HEAD 2^>nul') do set "SRC_COMMIT=%%H"
)
if not "!SRC_COMMIT!"=="%PROVIDER_COMMIT%" (
    echo Could not download the provider source ^(release %PROVIDER_TAG%^).
    echo Check your internet connection and run %RERUN% again.
    echo If you are behind a corporate proxy: git and npm honor the same
    echo HTTP_PROXY/HTTPS_PROXY/NO_PROXY variables as the rest of this installer
    echo - set them ^(and UV_SYSTEM_CERTS=1 for TLS-inspecting proxies^) and re-run.
    exit /b 3
)

REM --- Build it ---
REM The current build is renamed out of the way before anything is destroyed:
REM a rename fails as a whole while the folder is in use (e.g. by a running
REM provider, which works in it), where deleting it would leave half a build.
if exist "%SERVER_DIR%\build" if not exist "%SERVER_DIR%\build.old" move "%SERVER_DIR%\build" "%SERVER_DIR%\build.old" >nul 2>&1
if exist "%SERVER_DIR%\build" (
    echo The provider build folder %SERVER_DIR%\build is in use ^(close
    echo UltraSinger or end the node process that runs it, or wait for the virus
    echo scanner to finish^) - run %RERUN% again.
    exit /b 3
)
REM Start from scratch: a build of another release must neither be started by
REM the app nor make a failed build look successful (tsc's incremental state
REM would skip unchanged files, too). npm ci installs exactly the release's
REM package-lock.json, and tsc is the compiler it installed (npx --no never
REM downloads one). tsc writes into build.new, which only becomes build once
REM it is complete: an interrupted build never leaves half of one to start.
echo Building provider ^(npm ci + tsc, this can take a minute^)...
if exist "%SERVER_DIR%\build.new" rmdir /s /q "%SERVER_DIR%\build.new"
if exist "%SERVER_DIR%\tsconfig.tsbuildinfo" del /f /q "%SERVER_DIR%\tsconfig.tsbuildinfo"
set "BUILD_OK="
pushd "%SERVER_DIR%"
call npm ci --include=dev --no-audit --no-fund >nul 2>&1
if !errorlevel! equ 0 (
    call npx --no -- tsc --outDir build.new >nul 2>&1
    if !errorlevel! equ 0 if not exist build (
        move build.new build >nul 2>&1
        if exist build set "BUILD_OK=1"
    )
)
if exist build.new rmdir /s /q build.new
if exist build.old rmdir /s /q build.old
popd
if not exist "%SERVER_ENTRY%" set "BUILD_OK="

if defined BUILD_OK (
    >"%BUILD_STAMP%" echo %PROVIDER_COMMIT%
    REM The GUI's first-launch bootstrap starts the server itself right
    REM after this script, so the warm-up below would only double the wait.
    if defined ULTRASINGER_POTOKEN_SKIP_WARMUP (
        echo Provider built. Warm-up skipped ^(the app manages the server^).
        exit /b 0
    )
    REM Warm up the provider now so the FIRST app launch is fast. On first
    REM start, security software scans the freshly built node_modules, which
    REM on corporate machines can take several minutes - during which the app
    REM would look broken. Trigger that scan here instead.
    echo Warming up the provider so the first app launch is fast...
    echo ^(On managed/corporate machines antivirus scans the new files on the
    echo  first start - this can take several minutes. You can leave it running.^)
    powershell -NoProfile -Command "$entry = (Resolve-Path '%SERVER_ENTRY%').Path; $p = Start-Process -FilePath 'node' -ArgumentList $entry -WorkingDirectory (Split-Path $entry) -WindowStyle Hidden -PassThru; $ready = $false; for ($i = 0; $i -lt 450; $i++) { try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:4416/ping' -TimeoutSec 3; if ($r.StatusCode -eq 200) { $ready = $true; break } } catch {}; Start-Sleep -Seconds 2 }; try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}; if ($ready) { Write-Host 'Provider warmed up - the GUI will start it quickly from now on.' } else { Write-Host 'Provider warm-up did not finish in time; the first app launch may still take a few minutes while security scanning completes.' }"
    echo Done. The GUI starts the provider automatically on launch -
    echo full-quality video downloads are enabled.
    exit /b 0
) else (
    echo The provider build did not complete. Run %RERUN% again;
    echo if it keeps failing, video downloads stay limited to 360p.
    echo If you are behind a corporate proxy: git and npm honor the same
    echo HTTP_PROXY/HTTPS_PROXY/NO_PROXY variables as the rest of this installer
    echo - set them ^(and UV_SYSTEM_CERTS=1 for TLS-inspecting proxies^) and re-run.
    exit /b 3
)
