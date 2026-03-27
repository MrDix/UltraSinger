#!/bin/bash
# Launch UltraSinger CLI on macOS

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if .venv exists (install script must be run first)
if [ ! -d ".venv" ]; then
    echo "============================================================"
    echo " ERROR: Virtual environment not found."
    echo "============================================================"
    echo ""
    echo " You need to run the installation script first:"
    echo ""
    echo "   install/CPU/macos_cpu.sh"
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
    echo ""
    echo " You can also specify the path with: --ffmpeg /path/to/ffmpeg"
    echo "============================================================"
    echo ""
fi

source .venv/bin/activate
cd src
echo "Starting UltraSinger..."
echo "Use: python UltraSinger.py -h  for help"
echo ""
python UltraSinger.py "$@"
