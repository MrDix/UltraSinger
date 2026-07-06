#!/bin/bash
# Ensure ffmpeg is available (required for all audio/video processing).
# Called by both auto_install.sh and update.sh so a single command always
# leaves the environment complete. Non-fatal: prints an ACTION REQUIRED note
# if ffmpeg is missing and returns 0 so the caller continues.

if command -v ffmpeg >/dev/null 2>&1; then
    exit 0
fi

echo ""
echo "------------------------------------------------------------"
echo " ACTION REQUIRED - ffmpeg is required for all audio/video"
echo " processing. UltraSinger will NOT work until this is done:"
echo "     macOS:          brew install ffmpeg"
echo "     Debian/Ubuntu:  sudo apt install ffmpeg"
echo "     Fedora:         sudo dnf install ffmpeg"
echo "     or download from https://www.ffmpeg.org/download.html"
echo "------------------------------------------------------------"
echo ""
exit 0
