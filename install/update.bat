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
REM
REM Every update also locks the newest yt-dlp release: video platforms change
REM their internals often, so the yt-dlp version pinned in uv.lock soon fails
REM to download. If that upgrade is not possible (e.g. offline), the pinned
REM version is kept.

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

REM --- Ensure ffmpeg is present (same check as the installer) ----------------
if exist "%~dp0helpers\ensure_ffmpeg.bat" (
    call "%~dp0helpers\ensure_ffmpeg.bat"
)

REM --- Detect whether this is a CUDA-protected install -----------------------
set "IS_CUDA="
git ls-files -v pyproject.toml 2>nul | findstr /b /c:"S" >nul 2>&1
if !errorlevel! equ 0 set "IS_CUDA=1"
findstr /c:"whl/cu" pyproject.toml >nul 2>&1
if !errorlevel! equ 0 set "IS_CUDA=1"
REM The CUDA variant of the PyTorch index (cu128 unless the user picked another
REM one, e.g. cu126 for an older driver), put back after the pull
set "CUDA_VARIANT="
for /f "delims=" %%V in ('powershell -NoProfile -Command "$m = [regex]::Match([IO.File]::ReadAllText('pyproject.toml'), 'whl/(cu[0-9]+)'); if ($m.Success) { $m.Groups[1].Value }"') do set "CUDA_VARIANT=%%V"
if not defined CUDA_VARIANT set "CUDA_VARIANT=cu128"

set "BACKUP_DIR="
set "KEEP_BACKUP="
set "PULLED="
if defined IS_CUDA (
    echo CUDA-protected install detected - lifting the pyproject/uv.lock
    echo protection for the update...
    set "BACKUP_DIR=%TEMP%\ultrasinger_update_%RANDOM%%RANDOM%"
    mkdir "!BACKUP_DIR!"
    copy /y pyproject.toml "!BACKUP_DIR!\" >nul
    copy /y uv.lock "!BACKUP_DIR!\" >nul
    git update-index --no-skip-worktree pyproject.toml 2>nul
    git update-index --no-skip-worktree uv.lock 2>nul
    REM Only the PyTorch index line is carried over to the new pyproject.toml:
    REM say so, and keep a copy, when the file has other local changes. A file
    REM put back by an update that failed after its pull is the version of the
    REM commit before that pull (ORIG_HEAD) plus the index line: no change.
    for /f "tokens=1,2" %%A in ('git diff --numstat -- pyproject.toml 2^>nul') do (
        if not "%%A %%B"=="1 1" set "KEEP_BACKUP=1"
    )
    if defined KEEP_BACKUP (
        git rev-parse -q --verify ORIG_HEAD >nul 2>&1
        if !errorlevel! equ 0 (
            set "BASE_CHANGED="
            for /f "tokens=1,2" %%A in ('git diff --numstat ORIG_HEAD -- pyproject.toml 2^>nul') do (
                if not "%%A %%B"=="1 1" set "BASE_CHANGED=1"
            )
            if not defined BASE_CHANGED set "KEEP_BACKUP="
        )
    )
    if defined KEEP_BACKUP (
        echo Note: pyproject.toml has local changes besides the CUDA index. The
        echo update replaces them; your previous file is kept as
        echo   !BACKUP_DIR!\pyproject.toml
    )
    git checkout -- pyproject.toml uv.lock
) else (
    REM The in-app yt-dlp update rewrites uv.lock; such a local change would
    REM make "git pull" fail as soon as the repository's uv.lock changes.
    REM It is not lost: the newest yt-dlp is locked again after the pull.
    git diff --quiet HEAD -- uv.lock 2>nul
    if !errorlevel! equ 1 (
        echo Discarding the local changes to uv.lock - the newest yt-dlp is
        echo locked again after the update.
        git checkout HEAD -- uv.lock
    )
)

echo Pulling latest changes...
git pull --ff-only
if !errorlevel! neq 0 (
    echo Error: git pull failed. Resolve the issue above and re-run.
    goto :restore_and_fail
)
set "PULLED=1"

if defined IS_CUDA (
    echo Re-applying the CUDA PyTorch index ^(!CUDA_VARIANT!^)...
    powershell -NoProfile -Command "$c = [IO.File]::ReadAllText('pyproject.toml'); $c = $c -replace 'whl/cpu','whl/!CUDA_VARIANT!'; [IO.File]::WriteAllText('pyproject.toml', $c)"
)

echo Resolving dependencies with the newest yt-dlp...
uv lock --upgrade-package yt-dlp
if !errorlevel! neq 0 (
    echo Warning: could not upgrade yt-dlp - keeping the version from uv.lock.
    if defined IS_CUDA (
        uv lock
        if !errorlevel! neq 0 (
            echo Error during uv lock
            goto :restore_and_fail
        )
    )
)

REM Stop a running instance from this folder so uv can replace locked files.
REM The folder is handed over in an environment variable, so quotes in its
REM name cannot break the command.
set "US_REPO_ROOT=%CD%"
powershell -NoProfile -Command "$root = $env:US_REPO_ROOT; $venv = Join-Path $root '.venv'; foreach ($p in (Get-CimInstance Win32_Process | Where-Object { ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($venv, [System.StringComparison]::OrdinalIgnoreCase)) -or ($_.Name -eq 'node.exe' -and $_.CommandLine -and $_.CommandLine.ToLower().Contains('bgutil') -and $_.CommandLine.ToLower().Contains($root.ToLower())) })) { try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; Write-Host ('Closed running UltraSinger process (PID ' + $p.ProcessId + ').') } catch {} }"
if exist ".venv" timeout /t 2 /nobreak >nul

REM Avoid uv hardlink warnings when the cache and project are on different drives.
if not defined UV_LINK_MODE (
    set "UV_CACHE_PATH="
    for /f "delims=" %%D in ('uv cache dir 2^>nul') do set "UV_CACHE_PATH=%%D"
    if defined UV_CACHE_PATH (
        if /i not "!UV_CACHE_PATH:~0,1!"=="!CD:~0,1!" set "UV_LINK_MODE=copy"
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
    if not defined KEEP_BACKUP rmdir /s /q "!BACKUP_DIR!" 2>nul
)

REM Update the PO-token provider too, so a single "update" refreshes
REM everything (code, Python packages AND the provider) - the user never
REM needs to run the full installer just to pick up a provider change.
REM Non-fatal: provider setup exits non-zero when Node.js/git are missing.
if exist "%~dp0helpers\setup_potoken_provider.bat" (
    call "%~dp0helpers\setup_potoken_provider.bat" "install\update.bat"
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
    if not defined KEEP_BACKUP rmdir /s /q "!BACKUP_DIR!" 2>nul
)
if defined PULLED (
    echo.
    echo The update is incomplete: the code was updated, but the Python
    echo packages could not be brought up to date with it, so UltraSinger may
    echo fail to start or to work. Fix the problem shown above ^(e.g. no
    echo internet connection, proxy settings, files locked by a running
    echo UltraSinger^) and run install\update.bat again. If it keeps failing,
    echo install\auto_install.bat builds the environment anew.
)
pause
exit /b 1
