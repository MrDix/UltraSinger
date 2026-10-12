#!/bin/bash
set -e

cd "$(dirname "$0")"
cd ../..

# Set link mode to copy to avoid hardlink warnings on different filesystems
export UV_LINK_MODE=copy

# Install uv if not already installed (a fixed, reviewed release of its
# installer; uv can update itself later with "uv self update")
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/0.12.24/install.sh | sh
    # Update PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv could not be found or installed"
    echo "Please ensure your shell PATH includes ~/.local/bin"
    exit 1
fi

echo "uv version:"
uv --version

# Some dependencies are built from source: check for a C compiler and the
# Python headers now instead of failing deep in the build
bash install/helpers/check_build_tools.sh

# --- Keep the previous installation until the new one works -----------------
# pyproject.toml, uv.lock and .venv are only replaced once the new environment
# is built: if anything below fails (offline, a build error), the previous
# installation is restored and keeps working.
BACKUP_DIR="$(mktemp -d)"
cp -f pyproject.toml uv.lock "$BACKUP_DIR/"
WAS_PROTECTED=""
if git ls-files -v pyproject.toml 2>/dev/null | grep -q '^S'; then
    WAS_PROTECTED=1
fi
VENV_MOVED=""
restore_previous_install() {
    rc=$?
    set +e  # every restore step is best effort, none may end it early
    if [ "$rc" -ne 0 ]; then
        echo ""
        echo "Installation failed - restoring the previous installation..."
        if [ -n "$VENV_MOVED" ]; then
            # mv only into a free name: it would put .venv.old inside a leftover .venv
            if ! { rm -rf .venv && mv .venv.old .venv; }; then
                echo "The previous environment could not be put back and is kept in"
                echo ".venv.old - rename it to .venv once no program uses the folder."
            fi
        fi
        cp -f "$BACKUP_DIR/pyproject.toml" "$BACKUP_DIR/uv.lock" .
        if [ -n "$WAS_PROTECTED" ]; then
            git update-index --skip-worktree pyproject.toml uv.lock 2>/dev/null || true
        fi
    fi
    rm -rf "$BACKUP_DIR"
    # keep the script's own exit status (the trap's last command would set it)
    exit "$rc"
}
trap restore_previous_install EXIT

# Set PyTorch index to CUDA in pyproject.toml
# (uv.toml cannot override named indexes used by [tool.uv.sources])
echo "Configuring PyTorch index for CUDA..."
sed -i 's|whl/cpu|whl/cu128|' pyproject.toml

# Regenerate lockfile with CUDA PyTorch index and the newest yt-dlp (video
# platforms change often, so the yt-dlp pinned in uv.lock soon fails to
# download; if the upgrade is not possible, e.g. offline, keep the pin), then sync
echo "Resolving dependencies..."
if ! uv lock --upgrade-package yt-dlp; then
    echo "Warning: could not upgrade yt-dlp - keeping the version from uv.lock."
    uv lock
fi

# Build the environment from scratch (clean state, e.g. when switching between
# CPU and CUDA); the previous one waits in .venv.old until the new one is done.
if [ -d .venv.old ]; then
    # Left over from an interrupted run: keep .venv only if it was completed
    # (it has the marker written below), else go back to the previous one
    if [ -f .venv/.ultrasinger-complete ]; then
        rm -rf .venv.old
    else
        rm -rf .venv
        mv .venv.old .venv
    fi
fi
if [ -d .venv ]; then
    mv .venv .venv.old
    VENV_MOVED=1
fi
echo "Syncing dependencies (core + GUI + scoring + PO-token plugin)..."
uv sync --extra gui --extra scoring --extra potoken
# The new environment is complete: mark it, nothing is restored from here on
touch .venv/.ultrasinger-complete
trap - EXIT
rm -rf .venv.old "$BACKUP_DIR"

# Set up the PO-token provider (Node.js) for full-quality video downloads
POT_RC=0
bash install/helpers/setup_potoken_provider.sh "install/CUDA/linux_cuda_gpu.sh" || POT_RC=$?

# Protect local CUDA config from being reverted by git operations
# (branch switches, pulls, etc. would otherwise reset to CPU default)
if command -v git &> /dev/null && git rev-parse --is-inside-work-tree &> /dev/null; then
    echo "Protecting CUDA configuration from git resets..."
    git update-index --skip-worktree pyproject.toml
    git update-index --skip-worktree uv.lock
fi

echo ""
echo "Installation completed."
if [ "$POT_RC" = "0" ]; then
    echo "Full-quality video downloads are enabled."
else
    echo "NOTE: full-quality video downloads are NOT enabled yet -"
    echo "see the PO-token provider section above for what to do."
fi
echo ""
echo "To run UltraSinger:"
echo "  source .venv/bin/activate"
echo "  cd src"
echo "  python UltraSinger.py"
