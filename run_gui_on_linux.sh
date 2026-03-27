#!/bin/bash
# Launch UltraSinger GUI on Linux

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
    echo "   install/CPU/linux_cpu.sh       (CPU-only)"
    echo "   install/CUDA/linux_cuda_gpu.sh (NVIDIA GPU with CUDA)"
    echo ""
    echo " The install script will install uv automatically."
    echo " Or install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo " Prerequisites:"
    echo "   - Python 3.12 or 3.13  https://www.python.org/downloads/"
    echo "   - ffmpeg in PATH       https://www.ffmpeg.org/download.html"
    echo "     (or: sudo apt install ffmpeg)"
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
    echo " Install: sudo apt install ffmpeg  (Debian/Ubuntu)"
    echo "          sudo dnf install ffmpeg   (Fedora)"
    echo " Or download: https://www.ffmpeg.org/download.html"
    echo "============================================================"
    echo ""
fi

# Suppress compile-time SyntaxWarnings from third-party packages (e.g. pydub)
export PYTHONWARNINGS="ignore::SyntaxWarning"
echo "Starting UltraSinger GUI..."
uv run --extra gui python src/gui_main.py
