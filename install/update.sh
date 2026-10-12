#!/bin/bash
# Update an existing UltraSinger installation.
#
# Handles the CUDA case where the installer protects pyproject.toml and
# uv.lock with git skip-worktree: a plain "git pull" refuses to update
# those files. This script backs the protected files up, temporarily lifts
# the protection, pulls, re-applies the CUDA configuration and restores the
# protection - then syncs dependencies (without rebuilding the venv from
# scratch). If any step fails, the backed-up files and the protection are
# restored, so a failed update never leaves a CUDA install half-converted
# to CPU.
#
# Every update also locks the newest yt-dlp release: video platforms change
# their internals often, so the yt-dlp version pinned in uv.lock soon fails to
# download. If that upgrade is not possible (e.g. offline), the pinned version
# is kept.

set -e
# Capture the script directory BEFORE cd'ing away, so helper paths resolve
# correctly regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "Current directory: $(pwd)"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v git >/dev/null 2>&1; then
    echo "Error: git is required to update."
    exit 1
fi

# --- Ensure ffmpeg is present (same check as the installer) ------------------
if [ -f "$SCRIPT_DIR/helpers/ensure_ffmpeg.sh" ]; then
    bash "$SCRIPT_DIR/helpers/ensure_ffmpeg.sh" || true
fi

# --- Detect whether this is a CUDA-protected install -------------------------
IS_CUDA=""
if git ls-files -v pyproject.toml 2>/dev/null | grep -q '^S'; then
    IS_CUDA=1
fi
if grep -q 'whl/cu' pyproject.toml 2>/dev/null; then
    IS_CUDA=1
fi
# The CUDA variant of the PyTorch index (cu128 unless the user picked another
# one, e.g. cu126 for an older driver), put back after the pull
CUDA_VARIANT="$(grep -o 'whl/cu[0-9][0-9]*' pyproject.toml 2>/dev/null | head -n1 | cut -d/ -f2)"
CUDA_VARIANT="${CUDA_VARIANT:-cu128}"

# --- Failure safety: back up the protected files, restore them on any error --
BACKUP_DIR=""
KEEP_BACKUP=""
PULLED=""
cleanup() {
    rc=$?
    if [ "$rc" -ne 0 ] && [ -n "$IS_CUDA" ] && [ -n "$BACKUP_DIR" ]; then
        echo ""
        echo "Update failed - restoring the previous CUDA configuration..."
        cp -f "$BACKUP_DIR/pyproject.toml" pyproject.toml 2>/dev/null || true
        cp -f "$BACKUP_DIR/uv.lock" uv.lock 2>/dev/null || true
        git update-index --skip-worktree pyproject.toml 2>/dev/null || true
        git update-index --skip-worktree uv.lock 2>/dev/null || true
    fi
    if [ "$rc" -ne 0 ] && [ -n "$PULLED" ]; then
        echo ""
        echo "The update is incomplete: the code was updated, but the Python"
        echo "packages could not be brought up to date with it, so UltraSinger may"
        echo "fail to start or to work. Fix the problem shown above (e.g. no"
        echo "internet connection, proxy settings, files locked by a running"
        echo "UltraSinger) and run install/update.sh again. If it keeps failing,"
        echo "install/auto_install.sh builds the environment anew."
    fi
    if [ -n "$BACKUP_DIR" ] && [ -z "$KEEP_BACKUP" ]; then
        rm -rf "$BACKUP_DIR"
    fi
    # keep the script's own exit status (the trap's last command would set it)
    exit "$rc"
}
trap cleanup EXIT

if [ -n "$IS_CUDA" ]; then
    echo "CUDA-protected install detected - lifting the pyproject/uv.lock"
    echo "protection for the update..."
    BACKUP_DIR="$(mktemp -d)"
    cp -f pyproject.toml uv.lock "$BACKUP_DIR/"
    git update-index --no-skip-worktree pyproject.toml 2>/dev/null || true
    git update-index --no-skip-worktree uv.lock 2>/dev/null || true
    # Only the PyTorch index line is carried over to the new pyproject.toml:
    # say so, and keep a copy, when the file has other local changes. A file
    # put back by an update that failed after its pull is the version of the
    # commit before that pull (ORIG_HEAD) plus the index line: no change.
    only_the_index_differs() {  # $@ = the commit to compare with (default: the index)
        local changes
        changes="$(git diff --numstat "$@" -- pyproject.toml 2>/dev/null | cut -f1,2)"
        [ -z "$changes" ] || [ "$changes" = "$(printf '1\t1')" ]
    }
    if ! only_the_index_differs \
            && ! { git rev-parse -q --verify ORIG_HEAD >/dev/null 2>&1 && only_the_index_differs ORIG_HEAD; }; then
        KEEP_BACKUP=1
        echo "Note: pyproject.toml has local changes besides the CUDA index. The"
        echo "update replaces them; your previous file is kept as"
        echo "  $BACKUP_DIR/pyproject.toml"
    fi
    git checkout -- pyproject.toml uv.lock
else
    # The in-app yt-dlp update rewrites uv.lock; such a local change would make
    # "git pull" fail as soon as the repository's uv.lock changes. It is not
    # lost: the newest yt-dlp is locked again after the pull.
    LOCK_DIFF=0
    git diff --quiet HEAD -- uv.lock 2>/dev/null || LOCK_DIFF=$?
    if [ "$LOCK_DIFF" -eq 1 ]; then
        echo "Discarding the local changes to uv.lock - the newest yt-dlp is"
        echo "locked again after the update."
        git checkout HEAD -- uv.lock
    fi
fi

echo "Pulling latest changes..."
git pull --ff-only
PULLED=1

if [ -n "$IS_CUDA" ]; then
    echo "Re-applying the CUDA PyTorch index ($CUDA_VARIANT)..."
    # BSD/macOS sed needs a suffix argument for -i
    sed -i.bak "s|whl/cpu|whl/$CUDA_VARIANT|" pyproject.toml && rm -f pyproject.toml.bak
fi

echo "Resolving dependencies with the newest yt-dlp..."
if ! uv lock --upgrade-package yt-dlp; then
    echo "Warning: could not upgrade yt-dlp - keeping the version from uv.lock."
    if [ -n "$IS_CUDA" ]; then
        uv lock
    fi
fi

# Stop a running instance from this folder so uv can replace locked files.
if command -v pkill >/dev/null 2>&1; then
    pkill -f "$(pwd)/.venv" >/dev/null 2>&1 || true
    pkill -f "$(pwd)/.potoken" >/dev/null 2>&1 || true
    sleep 1
fi

# Avoid uv hardlink warnings when the cache and project are on different filesystems.
if [ -z "${UV_LINK_MODE:-}" ]; then
    CACHE_DIR="$(uv cache dir 2>/dev/null)"
    if [ -n "$CACHE_DIR" ] && [ -d "$CACHE_DIR" ]; then
        _probe_src="$CACHE_DIR/.us_linkprobe.$$"
        _probe_dst="$(pwd)/.us_linkprobe.$$"
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

echo "Syncing dependencies..."
uv sync --extra gui --extra scoring --extra potoken

if [ -n "$IS_CUDA" ]; then
    echo "Restoring the CUDA configuration protection..."
    git update-index --skip-worktree pyproject.toml
    git update-index --skip-worktree uv.lock
fi

# Update the PO-token provider too, so a single "update" refreshes everything
# (code, Python packages AND the provider) - the user never needs to run the
# full installer just to pick up a provider change. Non-fatal.
SETUP_HELPER="$SCRIPT_DIR/helpers/setup_potoken_provider.sh"
if [ -f "$SETUP_HELPER" ]; then
    bash "$SETUP_HELPER" "install/update.sh" || true
fi

echo ""
echo "Update completed."
