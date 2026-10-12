@echo off
setlocal enabledelayedexpansion

:: Set link mode to copy to avoid hardlink warnings on different filesystems
set UV_LINK_MODE=copy

:: Navigate to project root
pushd "%~dp0"
cd /d ..\..
echo Current directory: %cd%

:: Update PATH to include uv installation directory
set "PATH=%USERPROFILE%\.local\bin;!PATH!"

:: Find a system Python 3.13 or 3.12. Only its version is passed on to uv
:: (uv finds the interpreter by it), never its path: a path read back from
:: Python's output is garbled for user folders with non-ASCII characters
:: (Python writes the ANSI code page, cmd reads the OEM one).
set "PYTHON_VERSION="

for %%V in (3.13 3.12) do (
    py -%%V --version >nul 2>&1
    if !errorlevel! equ 0 (
        REM The launcher may resolve the tag to a different interpreter than
        REM requested (e.g. a stale registry entry left behind by an upgrade
        REM or uninstall), so verify the version before accepting it.
        set "PY_VER="
        for /f "delims=" %%W in ('py -%%V -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2^>nul') do (
            set "PY_VER=%%W"
        )
        if "!PY_VER!"=="%%V" (
            set "PYTHON_VERSION=%%V"
            goto :found_python
        )
    )
)

:: Fallback to direct Python installations (verify version before accepting)
for %%P in (python3.13 python3.12 python3 python) do (
    where %%P >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_VER="
        for /f "delims=" %%V in ('%%P -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2^>nul') do (
            set "PY_VER=%%V"
        )
        if "!PY_VER!"=="3.13" (
            set "PYTHON_VERSION=3.13"
            goto :found_python
        )
        if "!PY_VER!"=="3.12" (
            set "PYTHON_VERSION=3.12"
            goto :found_python
        )
    )
)

:found_python
if "!PYTHON_VERSION!"=="" (
    echo No system Python 3.12 or 3.13 found ^(e.g. only a newer version is installed^).
    echo A portable Python 3.12 will be downloaded by uv into its per-user
    echo directory instead - fully self-contained, no PATH/registry changes,
    echo your system Python stays untouched.
    set "USE_MANAGED_PYTHON=1"
) else (
    echo Using Python !PYTHON_VERSION! found on this system.
)

:: Install uv if not already installed (a fixed, reviewed release of its
:: installer; uv can update itself later with "uv self update")
where uv >nul 2>&1
if !errorlevel! neq 0 (
    echo Installing uv...
    REM -ExecutionPolicy Bypass applies to this one PowerShell process only
    REM (the system-wide policy is untouched): the uv install script refuses
    REM to run under the Windows default "Restricted" policy, and fresh
    REM machines would otherwise abort here with a manual
    REM "Set-ExecutionPolicy" instruction.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/0.12.24/install.ps1 | iex"
)

:: Wait a moment for uv to be available
timeout /t 2 /nobreak >nul

:: Verify uv is available
where uv >nul 2>&1
if !errorlevel! neq 0 (
    echo Error: uv could not be found or installed.
    echo If the message above mentions the PowerShell execution policy, that
    echo policy is enforced system-wide on this machine ^(common on managed
    echo devices^) and blocks the uv install script. Install uv manually
    echo instead, e.g. with winget:
    echo   winget install astral-sh.uv
    echo and then re-run this script.
    pause
    exit /b 1
)

echo uv is ready
uv --version

:: Fallback: no suitable system Python - let uv provide a managed one
if defined USE_MANAGED_PYTHON (
    echo Downloading portable Python 3.12 via uv ...
    uv python install 3.12
    if !errorlevel! neq 0 (
        echo First attempt failed - retrying with an app-local Python store
        echo ^(.uv-python inside this folder^). This sidesteps a broken or
        echo locked global uv Python store, e.g. from other uv-managed
        echo Python versions on this machine.
        set "UV_PYTHON_INSTALL_DIR=%cd%\.uv-python"
        uv python install 3.12
        if !errorlevel! neq 0 (
            echo Error: could not download a managed Python 3.12 via uv.
            echo If you are behind a corporate proxy, set HTTP_PROXY/HTTPS_PROXY
            echo and re-run. Alternatively install Python 3.12 or 3.13 manually.
            pause
            exit /b 1
        )
    )
    set "PYTHON_VERSION=3.12"
)

:: --- Keep the previous installation until the new one works -----------------
:: pyproject.toml, uv.lock and .venv are only replaced once the new environment
:: is built: if anything below fails (offline, a build error, files in use),
:: the previous installation is restored and keeps working.
set "BACKUP_DIR=%TEMP%\ultrasinger_install_%RANDOM%%RANDOM%"
mkdir "!BACKUP_DIR!" >nul 2>&1
copy /y pyproject.toml "!BACKUP_DIR!\" >nul
copy /y uv.lock "!BACKUP_DIR!\" >nul
set "WAS_PROTECTED="
git ls-files -v pyproject.toml 2>nul | findstr /b /c:"S" >nul 2>&1
if !errorlevel! equ 0 set "WAS_PROTECTED=1"
set "VENV_MOVED="
set "NEW_VENV="

:: Clear skip-worktree flags if previously set by CUDA install script
:: (allows git to manage these files normally again since CPU is the default)
where git >nul 2>&1
if !errorlevel! equ 0 (
    git update-index --no-skip-worktree pyproject.toml 2>nul
    git update-index --no-skip-worktree uv.lock 2>nul
)

:: Ensure PyTorch index is set to CPU in pyproject.toml
:: (restores default if previously changed by CUDA install script)
:: Use .NET WriteAllText to avoid UTF-8 BOM that breaks TOML parsing
:: (PowerShell 5.x Set-Content -Encoding UTF8 adds a BOM)
powershell -NoProfile -Command "$c = [IO.File]::ReadAllText('pyproject.toml'); $c = $c -replace 'whl/cu\d+','whl/cpu'; [IO.File]::WriteAllText('pyproject.toml', $c)"

:: Regenerate lockfile with CPU PyTorch index and the newest yt-dlp
:: (video platforms change often, so the yt-dlp pinned in uv.lock soon fails
:: to download; if the upgrade is not possible, e.g. offline, keep the pin)
echo Resolving dependencies...
uv lock --upgrade-package yt-dlp
if !errorlevel! neq 0 (
    echo Warning: could not upgrade yt-dlp - keeping the version from uv.lock.
    uv lock
    if !errorlevel! neq 0 (
        echo Error during uv lock
        goto :restore_and_fail
    )
)

:: Build the environment from scratch (clean state, e.g. when switching between
:: CPU and CUDA); the previous one waits in .venv.old until the new one is done.
if exist ".venv.old\" (
    REM Left over from an interrupted run: keep .venv only if it was completed
    REM (it has the marker written below), else go back to the previous one
    if exist ".venv\.ultrasinger-complete" (
        rmdir /s /q ".venv.old"
    ) else (
        if exist ".venv\" rmdir /s /q ".venv"
        if not exist ".venv\" move ".venv.old" ".venv" >nul
    )
)
if exist ".venv\" (
    if not exist ".venv.old\" move ".venv" ".venv.old" >nul 2>&1
    if exist ".venv\" (
        echo.
        if exist ".venv.old\" if exist ".venv\.ultrasinger-complete" (
            echo Error: .venv.old, the backup of an environment left behind by an
            echo interrupted installation, is in use and cannot be removed. Close
            echo UltraSinger and any terminal or editor that uses it, then run the
            echo installer again.
            goto :restore_and_fail
        )
        echo Error: the existing virtual environment ^(.venv^) is in use and cannot
        echo be replaced. Close UltraSinger and any terminal or editor that uses
        echo it, then run the installer again.
        goto :restore_and_fail
    )
    set "VENV_MOVED=1"
) else (
    REM No environment to put back if this run fails: the incomplete one it
    REM leaves behind is removed instead
    set "NEW_VENV=1"
)

echo Syncing dependencies (core + GUI + scoring + PO-token plugin)...
uv sync --python "!PYTHON_VERSION!" --extra gui --extra scoring --extra potoken
if !errorlevel! neq 0 (
    echo Error during uv sync
    goto :restore_and_fail
)
:: The new environment is complete: mark it, nothing is restored from here on
type nul > ".venv\.ultrasinger-complete"
if exist ".venv.old\" rmdir /s /q ".venv.old"
rmdir /s /q "!BACKUP_DIR!" 2>nul

REM Set up the PO-token provider (Node.js) for full-quality video downloads
call install\helpers\setup_potoken_provider.bat "install\CPU\windows_cpu.bat"
set "POT_RC=!errorlevel!"

echo.
echo Installation completed.
if "!POT_RC!"=="0" (
    echo Full-quality video downloads are enabled.
) else (
    echo NOTE: full-quality video downloads are NOT enabled yet -
    echo see the PO-token provider section above for what to do.
)
echo.
pause
exit /b 0

:restore_and_fail
echo.
echo Installation failed - restoring the previous installation...
if defined VENV_MOVED (
    if exist ".venv\" rmdir /s /q ".venv"
    REM only into a free name - move would put it inside a leftover .venv
    if not exist ".venv\" move ".venv.old" ".venv" >nul 2>&1
    if exist ".venv.old\" (
        echo The previous environment could not be put back and is kept in
        echo .venv.old - rename it to .venv once no program uses the folder.
    )
)
REM A first installation: the launchers would try to use a half-built .venv
if defined NEW_VENV if exist ".venv\" (
    rmdir /s /q ".venv"
    if exist ".venv\" (
        echo The incomplete environment could not be removed - delete the .venv
        echo folder once no program uses it.
    )
)
copy /y "!BACKUP_DIR!\pyproject.toml" pyproject.toml >nul 2>&1
copy /y "!BACKUP_DIR!\uv.lock" uv.lock >nul 2>&1
if defined WAS_PROTECTED (
    git update-index --skip-worktree pyproject.toml uv.lock >nul 2>&1
)
rmdir /s /q "!BACKUP_DIR!" 2>nul
pause
exit /b 1
