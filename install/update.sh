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

set -e
cd "$(dirname "$0")/.."
echo "Current directory: $(pwd)"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v git >/dev/null 2>&1; then
    echo "Error: git is required to update."
    exit 1
fi

# --- Detect whether this is a CUDA-protected install -------------------------
IS_CUDA=""
if git ls-files -v pyproject.toml 2>/dev/null | grep -q '^S'; then
    IS_CUDA=1
fi
if grep -q 'whl/cu' pyproject.toml 2>/dev/null; then
    IS_CUDA=1
fi

# --- Failure safety: back up the protected files, restore them on any error --
BACKUP_DIR=""
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
    [ -n "$BACKUP_DIR" ] && rm -rf "$BACKUP_DIR"
}
trap cleanup EXIT

if [ -n "$IS_CUDA" ]; then
    echo "CUDA-protected install detected - lifting the pyproject/uv.lock"
    echo "protection for the update..."
    BACKUP_DIR="$(mktemp -d)"
    cp -f pyproject.toml uv.lock "$BACKUP_DIR/"
    git update-index --no-skip-worktree pyproject.toml 2>/dev/null || true
    git update-index --no-skip-worktree uv.lock 2>/dev/null || true
    git checkout -- pyproject.toml uv.lock
fi

echo "Pulling latest changes..."
git pull --ff-only

if [ -n "$IS_CUDA" ]; then
    echo "Re-applying the CUDA PyTorch index..."
    # BSD/macOS sed needs a suffix argument for -i
    sed -i.bak 's|whl/cpu|whl/cu128|' pyproject.toml && rm -f pyproject.toml.bak
    uv lock
fi

echo "Syncing dependencies..."
uv sync --extra gui --extra scoring --extra potoken

if [ -n "$IS_CUDA" ]; then
    echo "Restoring the CUDA configuration protection..."
    git update-index --skip-worktree pyproject.toml
    git update-index --skip-worktree uv.lock
fi

echo ""
echo "Update completed."
