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
    set "PULLED="
    set "KEEP_BACKUP="
    goto :restore_and_fail
)

REM ---------------------------------------------------------------------------
REM Keep everything above, down to and including the "git pull --ff-only" line,
REM byte for byte as it is. cmd.exe reads a running batch file piece by piece:
REM when the pull replaces this file, the update.bat that is running (the one
REM from before the pull) carries on at the same byte offset in the new file -
REM with the same beginning, that is the line right after the pull. Add new
REM steps below this point only, and give every variable they use its value
REM down here: of those, an earlier update.bat only sets IS_CUDA and BACKUP_DIR
REM before its pull. pytest/test_install_scripts.py checks that the beginning
REM stays unchanged.
REM ---------------------------------------------------------------------------
set "PULLED=1"
set "KEEP_BACKUP="
set "CUDA_VARIANT="
if defined IS_CUDA call :read_previous_pyproject
if not defined CUDA_VARIANT set "CUDA_VARIANT=cu128"

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
REM Started from the GUI's update question: start the GUI again (it quit for
REM the update). A failed update stops at its pause above instead, so the
REM error stays readable. The launcher is named relative to the checkout (the
REM current folder): a full path in this block breaks on a ")" in it.
if defined ULTRASINGER_RESTART_GUI (
    set "ULTRASINGER_RESTART_GUI="
    echo Starting UltraSinger again...
    start "UltraSinger" cmd /c .\run_gui_on_windows.bat
    exit /b 0
)
pause
exit /b 0

:read_previous_pyproject
REM The pyproject.toml of the installation, as copied before the pull. It gives
REM the CUDA variant of the PyTorch index to put back - cu128 unless the user
REM picked another one, e.g. cu126 for an older driver.
if not defined BACKUP_DIR exit /b 0
if not exist "!BACKUP_DIR!\pyproject.toml" exit /b 0
set "US_PREVIOUS_PYPROJECT=!BACKUP_DIR!\pyproject.toml"
for /f "delims=" %%V in ('powershell -NoProfile -Command "$m = [regex]::Match([IO.File]::ReadAllText($env:US_PREVIOUS_PYPROJECT), 'whl/(cu[0-9]+)'); if ($m.Success) { $m.Groups[1].Value }"') do set "CUDA_VARIANT=%%V"
REM Only the index line is carried over to the new pyproject.toml: say so, and
REM keep the copy, when the file had other local changes - that is, when it is
REM not the file of a recent commit with just the index line changed. Recent
REM commits are the one before this pull (ORIG_HEAD) and the two HEAD was on
REM before its latest moves: an update that failed after its pull put back
REM the file of the commit before that pull.
set "KEEP_BACKUP=1"
for %%R in (ORIG_HEAD HEAD@{1} HEAD@{2}) do if defined KEEP_BACKUP call :compare_with_commit %%R
del "!BACKUP_DIR!\pyproject.base" "!BACKUP_DIR!\numstat.txt" 2>nul
if defined KEEP_BACKUP (
    echo Note: pyproject.toml has local changes besides the CUDA index. The
    echo update replaces them; your previous file is kept as
    echo   !BACKUP_DIR!\pyproject.toml
)
exit /b 0

:compare_with_commit
REM Clears KEEP_BACKUP when the copy is the pyproject.toml of the given commit
REM with at most one line changed. git diff exits with 1 for differences and
REM for a file it cannot read alike, so only an identical file, exit code 0,
REM or a count of exactly one changed line clears it - anything else keeps it.
git show "%~1:pyproject.toml" > "!BACKUP_DIR!\pyproject.base" 2>nul || exit /b 0
git diff --no-index --numstat --ignore-cr-at-eol "!BACKUP_DIR!\pyproject.base" "!BACKUP_DIR!\pyproject.toml" > "!BACKUP_DIR!\numstat.txt" 2>nul
if !errorlevel! equ 0 set "KEEP_BACKUP="
for /f "usebackq tokens=1,2" %%A in ("!BACKUP_DIR!\numstat.txt") do if "%%A %%B"=="1 1" set "KEEP_BACKUP="
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
