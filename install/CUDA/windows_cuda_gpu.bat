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

:: Remove old virtual environment to ensure clean state (e.g., switching between CPU/CUDA)
if exist .venv (
    echo Removing old virtual environment...
    rmdir /s /q .venv
)

:: First, find Python using to get full path
set "PYTHON_EXE="

for %%V in (3.13 3.12) do (
    py -%%V --version >nul 2>&1
    if !errorlevel! equ 0 (
        :: Get the full path to the Python executable
        for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable)"') do (
            set "PYTHON_EXE=%%P"
        )
        goto :found_python
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
            set "PYTHON_EXE=%%P"
            goto :found_python
        )
        if "!PY_VER!"=="3.12" (
            set "PYTHON_EXE=%%P"
            goto :found_python
        )
    )
)

:found_python
if "!PYTHON_EXE!"=="" (
    echo No system Python 3.12 or 3.13 found ^(e.g. only a newer version is installed^).
    echo A portable Python 3.12 will be downloaded by uv into its per-user
    echo directory instead - fully self-contained, no PATH/registry changes,
    echo your system Python stays untouched.
    set "USE_MANAGED_PYTHON=1"
) else (
    echo Using Python: !PYTHON_EXE!
    "!PYTHON_EXE!" --version
)

:: Install uv if not already installed
where uv >nul 2>&1
if !errorlevel! neq 0 (
    echo Installing uv...
    powershell -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
)

:: Wait a moment for uv to be available
timeout /t 2 /nobreak >nul

:: Verify uv is available
where uv >nul 2>&1
if !errorlevel! neq 0 (
    echo Error: uv could not be found or installed
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
    set "PYTHON_EXE=3.12"
)

:: Set PyTorch index to CUDA in pyproject.toml
:: (uv.toml cannot override named indexes used by [tool.uv.sources])
echo Configuring PyTorch index for CUDA...
:: Use .NET WriteAllText to avoid UTF-8 BOM that breaks TOML parsing
:: (PowerShell 5.x Set-Content -Encoding UTF8 adds a BOM)
powershell -NoProfile -Command "$c = [IO.File]::ReadAllText('pyproject.toml'); $c = $c -replace 'whl/cpu','whl/cu128'; [IO.File]::WriteAllText('pyproject.toml', $c)"

:: Regenerate lockfile with CUDA PyTorch index
echo Resolving dependencies...
uv lock
if !errorlevel! neq 0 (
    echo Error during uv lock
    pause
    exit /b 1
)

echo Syncing dependencies (core + GUI + scoring + PO-token plugin)...
uv sync --python "!PYTHON_EXE!" --extra gui --extra scoring --extra potoken
if !errorlevel! neq 0 (
    echo Error during uv sync
    pause
    exit /b 1
)

REM Set up the PO-token provider (Node.js) for full-quality YouTube downloads
call install\helpers\setup_potoken_provider.bat "install\CUDA\windows_cuda_gpu.bat"
set "POT_RC=!errorlevel!"

:: Protect local CUDA config from being reverted by git operations
:: (branch switches, pulls, etc. would otherwise reset to CPU default)
where git >nul 2>&1
if !errorlevel! equ 0 (
    echo Protecting CUDA configuration from git resets...
    git update-index --skip-worktree pyproject.toml
    git update-index --skip-worktree uv.lock
)

echo.
echo Installation completed.
if "!POT_RC!"=="0" (
    echo Full-quality YouTube downloads are enabled.
) else (
    echo NOTE: full-quality YouTube downloads are NOT enabled yet -
    echo see the PO-token provider section above for what to do.
)
echo.
pause
