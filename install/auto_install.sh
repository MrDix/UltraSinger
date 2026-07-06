#!/bin/bash
# Hardware-aware installer entry point for Linux and macOS.
#
# Detects whether a usable NVIDIA GPU is present and runs the matching
# sub-script (install/CUDA/linux_cuda_gpu.sh or install/CPU/{linux,macos}_cpu.sh).
# The sub-scripts themselves are untouched and can still be run directly.
#
# Usage:
#   install/auto_install.sh              auto-detect hardware and install
#   install/auto_install.sh --cpu        force the CPU build
#   install/auto_install.sh --cuda       force the CUDA build (Linux only)
#   install/auto_install.sh --help       show this help
#
# Environment variable override (same effect as the flags above):
#   ULTRASINGER_BUILD=cpu|cuda install/auto_install.sh
#
# This script does not configure any API keys; it only prints information
# and suggested (not applied) command-line flags at the end.

set -e

print_usage() {
    echo "Usage: auto_install.sh [--cpu|--cuda] [--help]"
    echo ""
    echo "  --cpu     Force the CPU build, even if an NVIDIA GPU is detected."
    echo "  --cuda    Force the CUDA build, even if no NVIDIA GPU is detected"
    echo "            (not available on macOS)."
    echo "  --help    Show this help and exit."
    echo ""
    echo "Without a flag, hardware is auto-detected via nvidia-smi."
    echo "ULTRASINGER_BUILD=cpu|cuda in the environment has the same effect"
    echo "as the matching flag."
}

# --- Parse arguments ---------------------------------------------------------
FORCE_BUILD=""
for arg in "$@"; do
    case "$arg" in
        --cpu) FORCE_BUILD="cpu" ;;
        --cuda) FORCE_BUILD="cuda" ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            print_usage
            exit 1
            ;;
    esac
done

if [ -z "$FORCE_BUILD" ] && [ -n "$ULTRASINGER_BUILD" ]; then
    case "$ULTRASINGER_BUILD" in
        cpu|CPU) FORCE_BUILD="cpu" ;;
        cuda|CUDA) FORCE_BUILD="cuda" ;;
        *) echo "Ignoring unrecognized ULTRASINGER_BUILD='$ULTRASINGER_BUILD'" ;;
    esac
fi

# --- Parse a single "name, vram" CSV line from nvidia-smi --------------------
# nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
# e.g. "NVIDIA GeForce RTX 3060, 12288" -> PARSED_GPU_NAME / PARSED_GPU_VRAM
parse_nvidia_smi_line() {
    local line="$1"
    PARSED_GPU_NAME=$(echo "$line" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}')
    PARSED_GPU_VRAM=$(echo "$line" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
}

# --- OS detection --------------------------------------------------------------
OS_NAME="$(uname -s 2>/dev/null || echo unknown)"
IS_MACOS=false
case "$OS_NAME" in
    Darwin) IS_MACOS=true ;;
esac

# --- GPU detection -------------------------------------------------------------
# macOS never has a supported NVIDIA GPU (see README) - always use the CPU path
# there, with no error and no "install the driver" hint.
GPU_DETECTED=false
GPU_NAME=""
GPU_VRAM=""
NVIDIA_SMI_MISSING=false

if [ "$IS_MACOS" = false ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        SMI_OUTPUT="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)"
        if [ -n "$SMI_OUTPUT" ]; then
            parse_nvidia_smi_line "$SMI_OUTPUT"
            if [ -n "$PARSED_GPU_VRAM" ] && echo "$PARSED_GPU_VRAM" | grep -Eq '^[0-9]+$'; then
                GPU_DETECTED=true
                GPU_NAME="$PARSED_GPU_NAME"
                GPU_VRAM="$PARSED_GPU_VRAM"
            fi
        fi
    else
        NVIDIA_SMI_MISSING=true
    fi
fi

echo "=================================================================="
echo " UltraSinger installer - hardware detection"
echo "=================================================================="
if [ "$GPU_DETECTED" = true ]; then
    echo "Detected GPU: $GPU_NAME, ${GPU_VRAM} MB"
else
    echo "No NVIDIA GPU detected."
    if [ "$IS_MACOS" = false ] && [ "$NVIDIA_SMI_MISSING" = true ]; then
        echo "If you have an NVIDIA GPU, install its driver and re-run, or force with --cuda."
    fi
fi
echo ""

# --- Decide which build to install ----------------------------------------
if [ -n "$FORCE_BUILD" ]; then
    BUILD="$FORCE_BUILD"
    if [ "$BUILD" = "cuda" ] && [ "$IS_MACOS" = true ]; then
        echo "CUDA is not supported on macOS. Falling back to the CPU build."
        BUILD="cpu"
    fi
else
    if [ "$GPU_DETECTED" = true ]; then
        BUILD="cuda"
        if [ -t 0 ]; then
            printf "Detected %s, using CUDA build. Press C for CPU, Enter to continue: " "$GPU_NAME"
            read -r REPLY || REPLY=""
            case "$REPLY" in
                [Cc]*) BUILD="cpu" ;;
            esac
        fi
    else
        BUILD="cpu"
    fi
fi

# --- Ensure ffmpeg is available (required for all audio/video processing) ---
bash "$(cd "$(dirname "$0")" && pwd)/helpers/ensure_ffmpeg.sh"

# --- Pick and run the matching sub-script -----------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$BUILD" = "cuda" ]; then
    TARGET_SCRIPT="$SCRIPT_DIR/CUDA/linux_cuda_gpu.sh"
else
    if [ "$IS_MACOS" = true ]; then
        TARGET_SCRIPT="$SCRIPT_DIR/CPU/macos_cpu.sh"
    else
        TARGET_SCRIPT="$SCRIPT_DIR/CPU/linux_cpu.sh"
    fi
fi

echo "Selected build: $BUILD"
echo "Running: $TARGET_SCRIPT"
echo ""

# --- Auto-enable UV_SYSTEM_CERTS behind a detected corporate proxy ------------
# TLS-inspecting corporate proxies replace the server certificate with one
# signed by an internal CA that uv's bundled certificate store doesn't know
# about, so plain "uv sync"/"uv lock" fail with certificate errors. uv reads
# UV_SYSTEM_CERTS to fall back to the OS certificate store instead, but most
# users won't know this variable exists - so when a proxy is clearly
# configured via the environment (and the user hasn't explicitly opted out),
# enable it automatically for the sub-script.
if [ -z "$UV_SYSTEM_CERTS" ] && { [ -n "$HTTP_PROXY" ] || [ -n "$http_proxy" ] || [ -n "$HTTPS_PROXY" ] || [ -n "$https_proxy" ]; }; then
    export UV_SYSTEM_CERTS=1
    echo "Proxy detected (HTTP(S)_PROXY set) - enabling UV_SYSTEM_CERTS=1 so uv"
    echo "trusts certificates from the OS store (needed behind TLS-inspecting"
    echo "corporate proxies). Set UV_SYSTEM_CERTS=0 to opt out."
    echo ""
fi

# --- Stop a running UltraSinger instance from THIS folder --------------------
# A running GUI locks files under .venv that the sub-script deletes/replaces.
# Stop only processes whose command line references this repo's .venv or the
# bgutil provider under .potoken - never unrelated python/node processes. The
# installer shell itself runs install/auto_install.sh (not under those dirs).
if command -v pkill >/dev/null 2>&1; then
    _repo_root="$(cd "$SCRIPT_DIR/.." && pwd)"
    pkill -f "$_repo_root/.venv" >/dev/null 2>&1 || true
    pkill -f "$_repo_root/.potoken" >/dev/null 2>&1 || true
    sleep 1
fi

# --- Avoid uv hardlink warnings when cache and project differ ----------------
# uv hardlinks wheels from its cache into .venv; hardlinks only work within one
# filesystem. When the cache and the project are on different filesystems, uv
# prints a "Failed to hardlink ... falling back to full copy" warning on every
# sync. Probe once and set copy mode up front so the warning never appears;
# same-filesystem setups keep the faster hardlink path.
if [ -z "${UV_LINK_MODE:-}" ]; then
    CACHE_DIR="$(uv cache dir 2>/dev/null)"
    if [ -n "$CACHE_DIR" ] && [ -d "$CACHE_DIR" ]; then
        _probe_src="$CACHE_DIR/.us_linkprobe.$$"
        _probe_dst="$(cd "$SCRIPT_DIR/.." && pwd)/.us_linkprobe.$$"
        if : > "$_probe_src" 2>/dev/null; then
            if ln "$_probe_src" "$_probe_dst" 2>/dev/null; then
                rm -f "$_probe_dst"
            else
                export UV_LINK_MODE=copy
            fi
            rm -f "$_probe_src"
        fi
    fi
fi

set +e
bash "$TARGET_SCRIPT"
SUB_RC=$?
set -e

if [ "$SUB_RC" -ne 0 ]; then
    echo ""
    echo "Installation failed (exit code $SUB_RC). See the output above for details."
    echo ""
    echo "If you are behind a corporate proxy: set HTTP_PROXY/HTTPS_PROXY (and"
    echo "NO_PROXY), for TLS-inspecting proxies additionally set"
    echo "UV_SYSTEM_CERTS=1 (older uv: UV_NATIVE_TLS=1), then re-run."
    exit "$SUB_RC"
fi

# --- Final hardware-aware summary -------------------------------------------
echo ""
echo "=================================================================="
echo " Hardware summary"
echo "=================================================================="
if [ "$GPU_DETECTED" = true ]; then
    echo "Detected GPU: $GPU_NAME, ${GPU_VRAM} MB"
else
    echo "No NVIDIA GPU detected."
fi
echo ""

case "$BUILD" in
    cpu)
        echo "No NVIDIA GPU is being used for this install - CPU-only transcription"
        echo "can take several minutes per song."
        echo ""
        echo "Cost-saving tip: get a free API key at https://console.groq.com and run"
        echo "UltraSinger with --remote_stt (plus --remote_stt_api_key, or the"
        echo "ULTRASINGER_REMOTE_STT_API_KEY env var) to offload the slow transcription"
        echo "step to the cloud (a few seconds, free tier available); everything else"
        echo "still runs locally."
        echo "GUI users: enable this under Settings -> 'Remote Speech-to-Text'"
        echo "(paste the API key there; 'Fetch' lists the available models)."
        ;;
    cuda)
        if [ -z "$GPU_VRAM" ]; then
            echo "GPU VRAM could not be verified (forced CUDA build)."
            echo "If your GPU has less than 8 GB VRAM, the default Whisper model"
            echo "(large-v2) may run out of memory. Two independent ways to fit it"
            echo "(not set automatically):"
            echo "  --whisper_batch_size 4    Fewer segments in parallel: slower,"
            echo "                            but the transcription is UNCHANGED"
            echo "                            (the safe lever; lower to 2 or 1 if"
            echo "                            it still runs out)."
            echo "  --whisper_compute_type int8   Halves the model's memory at a"
            echo "                            small accuracy cost - add this only"
            echo "                            if lowering the batch size is not enough."
            echo "GUI users: Settings -> 'Transcription (Whisper)' -> lower"
            echo "'Batch Size' first, then set 'Compute Type' to int8 if needed."
            echo ""
            echo "Alternative: a free API key at https://console.groq.com plus --remote_stt"
            echo "runs transcription in the cloud instead of on your GPU."
            echo "GUI users: enable this under Settings -> 'Remote Speech-to-Text'."
        elif [ "$GPU_VRAM" -lt 8192 ]; then
            echo "Your GPU has less than 8 GB VRAM. The default Whisper model"
            echo "(large-v2) may run out of memory. Two independent ways to fit it"
            echo "(not set automatically):"
            echo "  --whisper_batch_size 4    Fewer segments in parallel: slower,"
            echo "                            but the transcription is UNCHANGED"
            echo "                            (the safe lever; lower to 2 or 1 if"
            echo "                            it still runs out)."
            echo "  --whisper_compute_type int8   Halves the model's memory at a"
            echo "                            small accuracy cost - add this only"
            echo "                            if lowering the batch size is not enough."
            echo "GUI users: Settings -> 'Transcription (Whisper)' -> lower"
            echo "'Batch Size' first, then set 'Compute Type' to int8 if needed."
            echo ""
            echo "Alternative: a free API key at https://console.groq.com plus --remote_stt"
            echo "runs transcription in the cloud instead of on your GPU (also saves VRAM)."
            echo "GUI users: enable this under Settings -> 'Remote Speech-to-Text'."
        else
            echo "All set, defaults are fine."
        fi
        ;;
esac
echo "=================================================================="
echo ""
echo "No API keys were configured automatically. See the tips above and"
echo "the README for how to set --remote_stt up if you want to use it"
echo "(GUI: Settings -> 'Remote Speech-to-Text')."
