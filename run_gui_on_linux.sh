#!/bin/bash
# Launch UltraSinger GUI on Linux

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed or not in PATH."
    echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "Starting UltraSinger GUI..."
uv run --extra gui python src/gui_main.py
