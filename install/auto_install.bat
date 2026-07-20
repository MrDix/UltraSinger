@echo off
setlocal enabledelayedexpansion
REM Hardware-aware installer entry point for Windows.
REM
REM Detects whether a usable NVIDIA GPU is present and runs the matching
REM sub-script (install\CUDA\windows_cuda_gpu.bat or install\CPU\windows_cpu.bat).
REM The sub-scripts themselves are untouched and can still be run directly.
REM
REM Usage:
REM   install\auto_install.bat              auto-detect hardware and install
REM   install\auto_install.bat --cpu        force the CPU build
REM   install\auto_install.bat --cuda       force the CUDA build
REM   install\auto_install.bat --help       show this help
REM
REM Environment variable override (same effect as the flags above):
REM   set ULTRASINGER_BUILD=cpu|cuda  (before running auto_install.bat)
REM
REM This script does not configure any API keys; it only prints information
REM and suggested (not applied) command-line flags at the end.

set "FORCE_BUILD="

for %%A in (%*) do (
    if /i "%%~A"=="--cpu" set "FORCE_BUILD=cpu"
    if /i "%%~A"=="--cuda" set "FORCE_BUILD=cuda"
    if /i "%%~A"=="--help" set "SHOW_HELP=1"
    if /i "%%~A"=="-h" set "SHOW_HELP=1"
    if /i "%%~A"=="/?" set "SHOW_HELP=1"
)

if defined SHOW_HELP (
    echo Usage: auto_install.bat [--cpu^|--cuda] [--help]
    echo.
    echo   --cpu     Force the CPU build, even if an NVIDIA GPU is detected.
    echo   --cuda    Force the CUDA build, even if no NVIDIA GPU is detected.
    echo   --help    Show this help and exit.
    echo.
    echo Without a flag, hardware is auto-detected via nvidia-smi.
    echo ULTRASINGER_BUILD=cpu^|cuda in the environment has the same effect
    echo as the matching flag.
    exit /b 0
)

if not defined FORCE_BUILD (
    if /i "%ULTRASINGER_BUILD%"=="cpu" set "FORCE_BUILD=cpu"
    if /i "%ULTRASINGER_BUILD%"=="cuda" set "FORCE_BUILD=cuda"
)

REM --- GPU detection -----------------------------------------------------------
REM Uses PowerShell to robustly parse the CSV line from nvidia-smi, e.g.
REM "NVIDIA GeForce RTX 3060, 12288" -> name="NVIDIA GeForce RTX 3060" vram=12288
set "GPU_DETECTED=0"
set "GPU_NAME="
set "GPU_VRAM="
set "NVIDIA_SMI_MISSING=0"
set "GPU_LINE="

where nvidia-smi >nul 2>&1
if !errorlevel! neq 0 (
    set "NVIDIA_SMI_MISSING=1"
) else (
    for /f "usebackq delims=" %%L in (`powershell -NoProfile -Command ^
        "$p = (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1) -split ','; if ($p.Count -ge 2) { Write-Output ($p[0].Trim() + '|' + $p[1].Trim()) }"`) do (
        set "GPU_LINE=%%L"
    )
    if defined GPU_LINE (
        for /f "tokens=1,2 delims=|" %%A in ("!GPU_LINE!") do (
            set "GPU_NAME=%%A"
            set "GPU_VRAM=%%B"
        )
        REM Validate VRAM is a plain integer before trusting it
        set "VRAM_CHECK=!GPU_VRAM!"
        if defined VRAM_CHECK (
            for /f "delims=0123456789" %%x in ("!VRAM_CHECK!") do set "VRAM_NONDIGIT=%%x"
            if not defined VRAM_NONDIGIT (
                set "GPU_DETECTED=1"
            ) else (
                set "GPU_NAME="
                set "GPU_VRAM="
            )
        )
    )
)

echo ==================================================================
echo  UltraSinger installer - hardware detection
echo ==================================================================
if "!GPU_DETECTED!"=="1" (
    echo Detected GPU: !GPU_NAME!, !GPU_VRAM! MB
) else (
    echo No NVIDIA GPU detected.
    if "!NVIDIA_SMI_MISSING!"=="1" (
        echo If you have an NVIDIA GPU, install its driver and re-run, or force with --cuda.
    )
)
echo.

REM --- Decide which build to install -------------------------------------------
set "BUILD="
if defined FORCE_BUILD (
    set "BUILD=!FORCE_BUILD!"
) else (
    if "!GPU_DETECTED!"=="1" (
        set "BUILD=cuda"
        rem Only prompt when stdin is an interactive console, so an automated
        rem or wrapped invocation cannot block waiting for input (mirrors the
        rem [ -t 0 ] guard in auto_install.sh). Non-interactive keeps the CUDA build.
        set "IS_INTERACTIVE=1"
        powershell -NoProfile -Command "if ([Console]::IsInputRedirected) { exit 1 } else { exit 0 }" >nul 2>&1
        if !errorlevel! neq 0 set "IS_INTERACTIVE=0"
        if "!IS_INTERACTIVE!"=="1" (
            set /p "REPLY=Detected !GPU_NAME!, using CUDA build. Press C for CPU, Enter to continue: "
            if /i "!REPLY:~0,1!"=="C" set "BUILD=cpu"
        )
    ) else (
        set "BUILD=cpu"
    )
)

REM --- Ensure ffmpeg is available (required for all audio/video processing) ---
call "%~dp0helpers\ensure_ffmpeg.bat"

REM --- Pick and run the matching sub-script ------------------------------------
if "!BUILD!"=="cuda" (
    set "TARGET_SCRIPT=%~dp0CUDA\windows_cuda_gpu.bat"
) else (
    set "TARGET_SCRIPT=%~dp0CPU\windows_cpu.bat"
)

echo Selected build: !BUILD!
echo Running: !TARGET_SCRIPT!
echo.

REM --- Auto-enable UV_SYSTEM_CERTS behind a detected corporate proxy -------------
REM TLS-inspecting corporate proxies replace the server certificate with one
REM signed by an internal CA that uv's bundled certificate store doesn't know
REM about, so plain "uv sync"/"uv lock" fail with certificate errors. uv reads
REM UV_SYSTEM_CERTS to fall back to the OS certificate store instead, but most
REM users won't know this variable exists - so when a proxy is clearly
REM configured via the environment (and the user hasn't explicitly opted out),
REM enable it automatically for the sub-script. (Windows env var names are
REM case-insensitive, so checking the lowercase forms too is just belt-and-
REM braces for shells/tools that set them that way.)
if not defined UV_SYSTEM_CERTS (
    set "PROXY_DETECTED="
    if defined HTTP_PROXY set "PROXY_DETECTED=1"
    if defined http_proxy set "PROXY_DETECTED=1"
    if defined HTTPS_PROXY set "PROXY_DETECTED=1"
    if defined https_proxy set "PROXY_DETECTED=1"
    if defined PROXY_DETECTED (
        set "UV_SYSTEM_CERTS=1"
        echo Proxy detected ^(HTTP^(S^)_PROXY set^) - enabling UV_SYSTEM_CERTS=1 so uv
        echo trusts certificates from the OS store ^(needed behind TLS-inspecting
        echo corporate proxies^). Set UV_SYSTEM_CERTS=0 to opt out.
        echo.
    )
)

REM --- Stop a running UltraSinger instance from THIS folder --------------------
REM A running GUI (python.exe + QtWebEngineProcess.exe under .venv) and the
REM bgutil Node provider lock files the sub-script needs to delete/replace,
REM causing "access denied" errors and a half-updated environment. Stop only
REM processes belonging to THIS repo (executables under .venv, or the bgutil
REM node), never unrelated python/node processes; the installer's own cmd and
REM PowerShell run from System32 and are not matched.
powershell -NoProfile -Command "$venv = Join-Path '%CD%' '.venv'; $root = '%CD%'; $procs = Get-CimInstance Win32_Process | Where-Object { ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($venv, [System.StringComparison]::OrdinalIgnoreCase)) -or ($_.Name -eq 'node.exe' -and $_.CommandLine -and $_.CommandLine.ToLower().Contains('bgutil') -and $_.CommandLine.ToLower().Contains($root.ToLower())) }; foreach ($p in $procs) { try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; Write-Host ('Closed running UltraSinger process (PID ' + $p.ProcessId + ').') } catch {} }"
REM Give Windows a moment to release the file handles before the venv is wiped.
if exist ".venv" timeout /t 2 /nobreak >nul

REM --- Avoid uv hardlink warnings when cache and project differ ----------------
REM uv hardlinks wheels from its cache into .venv; hardlinks only work within
REM one volume/filesystem. When the cache (often on C:) and the project are on
REM different drives, uv prints a "Failed to hardlink ... falling back to full
REM copy" warning on every sync. Detect that case and set copy mode up front so
REM the warning never appears; same-drive setups keep the faster hardlink path.
if not defined UV_LINK_MODE (
    set "UV_CACHE_PATH="
    for /f "delims=" %%D in ('uv cache dir 2^>nul') do set "UV_CACHE_PATH=%%D"
    if defined UV_CACHE_PATH (
        if /i not "!UV_CACHE_PATH:~0,1!"=="!CD:~0,1!" set "UV_LINK_MODE=copy"
    )
)

call "!TARGET_SCRIPT!"
set "SUB_RC=!errorlevel!"

if not "!SUB_RC!"=="0" (
    echo.
    echo Installation failed ^(exit code !SUB_RC!^). See the output above for details.
    echo.
    echo If you are behind a corporate proxy: set HTTP_PROXY/HTTPS_PROXY ^(and
    echo NO_PROXY^), for TLS-inspecting proxies additionally set
    echo UV_SYSTEM_CERTS=1 ^(older uv: UV_NATIVE_TLS=1^), then re-run.
    exit /b !SUB_RC!
)

REM --- Final hardware-aware summary --------------------------------------------
echo.
echo ==================================================================
echo  Hardware summary
echo ==================================================================
if "!GPU_DETECTED!"=="1" (
    echo Detected GPU: !GPU_NAME!, !GPU_VRAM! MB
) else (
    echo No NVIDIA GPU detected.
)
echo.

if "!BUILD!"=="cpu" (
    echo No NVIDIA GPU is being used for this install - CPU-only transcription
    echo can take several minutes per song.
    echo.
    echo Cost-saving tip: get a free API key at https://console.groq.com and run
    echo UltraSinger with --remote_stt ^(plus --remote_stt_api_key, or the
    echo ULTRASINGER_REMOTE_STT_API_KEY env var^) to offload the slow transcription
    echo step to the cloud ^(a few seconds, free tier available^); everything else
    echo still runs locally.
    echo GUI users: enable this under Settings -^> 'Remote Speech-to-Text'
    echo ^(paste the API key there; 'Fetch' lists the available models^).
) else (
    if not defined GPU_VRAM (
        echo GPU VRAM could not be verified ^(forced CUDA build^).
        echo UltraSinger scales the Whisper batch size automatically to the
        echo GPU memory it detects at runtime ^(16 on ~8 GB and larger, 8 on
        echo 5-7 GB, 4 on smaller cards^), which usually makes the default
        echo Whisper model ^(large-v2^) fit. If it still runs out of memory:
        echo   --whisper_batch_size 2    ^(or 1^) Fewer segments in parallel:
        echo                             slower, but the transcription is
        echo                             UNCHANGED ^(the safe lever^).
        echo   --whisper_compute_type int8    Halves the model's memory at a
        echo                             small accuracy cost - add this only if
        echo                             lowering the batch size is not enough.
        echo GUI users: Settings -^> 'Transcription ^(Whisper^)' -^> lower
        echo 'Batch Size' first, then set 'Compute Type' to int8 if needed.
        echo.
        echo Alternative: a free API key at https://console.groq.com plus --remote_stt
        echo runs transcription in the cloud instead of on your GPU.
        echo GUI users: enable this under Settings -^> 'Remote Speech-to-Text'.
    ) else (
        if !GPU_VRAM! LSS 8192 (
            echo Your GPU has less than 8 GB VRAM. UltraSinger scales the
            echo Whisper batch size automatically to the GPU memory ^(16 on
            echo ~8 GB and larger, 8 on 5-7 GB, 4 on smaller cards^), which
            echo usually makes the default Whisper model ^(large-v2^) fit.
            echo If it still runs out of memory:
            echo   --whisper_batch_size 2    ^(or 1^) Fewer segments in
            echo                             parallel: slower, but the
            echo                             transcription is UNCHANGED
            echo                             ^(the safe lever^).
            echo   --whisper_compute_type int8    Halves the model's memory at a
            echo                             small accuracy cost - add this only
            echo                             if lowering the batch size is not
            echo                             enough.
            echo GUI users: Settings -^> 'Transcription ^(Whisper^)' -^> lower
            echo 'Batch Size' first, then set 'Compute Type' to int8 if needed.
            echo.
            echo Alternative: a free API key at https://console.groq.com plus --remote_stt
            echo runs transcription in the cloud instead of on your GPU ^(also saves VRAM^).
            echo GUI users: enable this under Settings -^> 'Remote Speech-to-Text'.
        ) else (
            echo All set, defaults are fine.
        )
    )
)
echo ==================================================================
echo.
echo No API keys were configured automatically. See the tips above and
echo the README for how to set --remote_stt up if you want to use it
echo ^(GUI: Settings -^> 'Remote Speech-to-Text'^).
echo.

REM --- Optional: Desktop / Start Menu shortcuts for the GUI ---------------
REM Only ask when stdin is an interactive console (same guard as the
REM CPU/CUDA prompt above); automated runs skip the prompt and just print
REM how to create the shortcuts later.
powershell -NoProfile -Command "if ([Console]::IsInputRedirected) { exit 1 } else { exit 0 }" >nul 2>&1
if !errorlevel! equ 0 (
    set "REPLY="
    set /p "REPLY=Create Desktop and Start Menu shortcuts for the UltraSinger GUI? [Y/n]: "
    if /i not "!REPLY:~0,1!"=="n" (
        call "%~dp0create_desktop_shortcut.bat" nopause
    ) else (
        echo Skipped. You can create them any time by running
        echo   install\create_desktop_shortcut.bat
    )
) else (
    echo Tip: run install\create_desktop_shortcut.bat to create Desktop and
    echo Start Menu shortcuts for the GUI ^(with the UltraSinger icon^).
)
echo.
pause
