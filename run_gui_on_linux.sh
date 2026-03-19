#!/bin/bash
# Launch UltraSinger GUI on Linux

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found at .venv"
    echo "Please run one of the installation scripts first:"
    echo "  - install/CPU/linux_cpu.sh (for CPU)"
    echo "  - install/CUDA/linux_cuda_gpu.sh (for GPU with CUDA)"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Navigate to src and start UltraSinger GUI
cd src
echo "Starting UltraSinger GUI..."
python gui_main.py
