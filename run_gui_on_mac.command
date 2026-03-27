#!/bin/bash
# Launch UltraSinger GUI on macOS

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "============================================================"
    echo " ERROR: uv is not installed or not in PATH."
    echo "============================================================"
    echo ""
    echo " The GUI requires uv (Python package manager)."
    echo ""
    echo " You need to run the installation script first:"
    echo ""
    echo "   install/CPU/macos_cpu.sh"
    echo ""
    echo " The install script will install uv automatically."
    echo " Or install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo " Prerequisites:"
    echo "   - Python 3.12 or 3.13  https://www.python.org/downloads/"
    echo "   - ffmpeg in PATH       https://www.ffmpeg.org/download.html"
    echo "     (or: brew install ffmpeg)"
    echo ""
    echo " Note: macOS uses CPU-only mode (Apple Silicon does not"
    echo "       support NVIDIA CUDA)."
    echo ""
    echo " See README.md section \"How to use this source code\" for details."
    echo "============================================================"
    exit 1
fi

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "============================================================"
    echo " WARNING: ffmpeg not found in PATH."
    echo "============================================================"
    echo ""
    echo " UltraSinger requires ffmpeg for audio/video processing."
    echo " Install: brew install ffmpeg"
    echo " Or download: https://www.ffmpeg.org/download.html"
    echo "============================================================"
    echo ""
fi

# Suppress compile-time SyntaxWarnings from third-party packages (e.g. pydub)
export PYTHONWARNINGS="ignore::SyntaxWarning"
echo "Starting UltraSinger GUI..."
uv run --extra gui python src/gui_main.py
