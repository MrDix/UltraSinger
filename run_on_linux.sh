#!/bin/bash
# Launch UltraSinger CLI on Linux

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
    echo "   install/CPU/linux_cpu.sh       (CPU-only)"
    echo "   install/CUDA/linux_cuda_gpu.sh (NVIDIA GPU with CUDA)"
    echo ""
    echo " Prerequisites:"
    echo "   - Python 3.12 or 3.13  https://www.python.org/downloads/"
    echo "   - ffmpeg in PATH       https://www.ffmpeg.org/download.html"
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
