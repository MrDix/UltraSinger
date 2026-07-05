@echo off
setlocal enabledelayedexpansion
REM Update an existing UltraSinger installation.
REM
REM Handles the CUDA case where the installer protects pyproject.toml and
REM uv.lock with git skip-worktree: a plain "git pull" refuses to update
REM those files. This script backs the protected files up, temporarily
REM lifts the protection, pulls, re-applies the CUDA configuration and
REM restores the protection - then syncs dependencies (without rebuilding
REM the venv from scratch). If any step fails, the backed-up files and the
REM protection are restored, so a failed update never leaves a CUDA
REM install half-converted to CPU.

pushd "%~dp0"
cd /d ..
echo Current directory: %cd%
set "PATH=%USERPROFILE%\.local\bin;!PATH!"

where git >nul 2>&1
if !errorlevel! neq 0 (
    echo Error: git is required to update.
    pause
    exit /b 1
)

REM --- Detect whether this is a CUDA-protected install -----------------------
set "IS_CUDA="
git ls-files -v pyproject.toml 2>nul | findstr /b /c:"S" >nul 2>&1
if !errorlevel! equ 0 set "IS_CUDA=1"
findstr /c:"whl/cu" pyproject.toml >nul 2>&1
if !errorlevel! equ 0 set "IS_CUDA=1"

set "BACKUP_DIR="
if defined IS_CUDA (
    echo CUDA-protected install detected - lifting the pyproject/uv.lock
    echo protection for the update...
    set "BACKUP_DIR=%TEMP%\ultrasinger_update_%RANDOM%%RANDOM%"
    mkdir "!BACKUP_DIR!"
    copy /y pyproject.toml "!BACKUP_DIR!\" >nul
    copy /y uv.lock "!BACKUP_DIR!\" >nul
    git update-index --no-skip-worktree pyproject.toml 2>nul
    git update-index --no-skip-worktree uv.lock 2>nul
    git checkout -- pyproject.toml uv.lock
)

echo Pulling latest changes...
git pull --ff-only
if !errorlevel! neq 0 (
    echo Error: git pull failed. Resolve the issue above and re-run.
    goto :restore_and_fail
)

if defined IS_CUDA (
    echo Re-applying the CUDA PyTorch index...
    powershell -NoProfile -Command "$c = [IO.File]::ReadAllText('pyproject.toml'); $c = $c -replace 'whl/cpu','whl/cu128'; [IO.File]::WriteAllText('pyproject.toml', $c)"
    uv lock
    if !errorlevel! neq 0 (
        echo Error during uv lock
        goto :restore_and_fail
    )
)

echo Syncing dependencies...
uv sync --extra gui --extra scoring --extra potoken
if !errorlevel! neq 0 (
    echo Error during uv sync
    goto :restore_and_fail
)

if defined IS_CUDA (
    echo Restoring the CUDA configuration protection...
    git update-index --skip-worktree pyproject.toml
    git update-index --skip-worktree uv.lock
    rmdir /s /q "!BACKUP_DIR!" 2>nul
)

echo.
echo Update completed.
echo.
pause
exit /b 0

:restore_and_fail
if defined IS_CUDA (
    echo.
    echo Update failed - restoring the previous CUDA configuration...
    copy /y "!BACKUP_DIR!\pyproject.toml" pyproject.toml >nul 2>&1
    copy /y "!BACKUP_DIR!\uv.lock" uv.lock >nul 2>&1
    git update-index --skip-worktree pyproject.toml 2>nul
    git update-index --skip-worktree uv.lock 2>nul
    rmdir /s /q "!BACKUP_DIR!" 2>nul
)
pause
exit /b 1
