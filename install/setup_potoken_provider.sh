#!/bin/bash
# Set up the bgutil PO-token provider (Node.js server) so YouTube downloads
# get full-quality formats. Non-fatal: the parent install script keeps going
# regardless. Exit codes: 0 = provider ready, 2 = Node.js/git missing,
# 3 = build failed. Arg $1 = the install script to name in the re-run hint.
#
# Must be run from the repository root (the install scripts cd there first).

set +e  # never abort the parent install script

RERUN="${1:-the install script you just ran}"
PROVIDER_DIR=".potoken/bgutil-ytdlp-pot-provider"
PROVIDER_REPO="https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git"
SERVER_ENTRY="$PROVIDER_DIR/server/build/main.js"

echo ""
echo "============================================================"
echo " PO-token provider setup (full-quality YouTube downloads)"
echo "============================================================"

# --- Ensure Node.js / npm are available -------------------------------------
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    # Best-effort auto-install where a package manager is clearly available.
    if command -v brew >/dev/null 2>&1; then
        echo "Node.js not found. Installing via Homebrew..."
        brew install node >/dev/null 2>&1
    fi
fi

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo ""
    echo "------------------------------------------------------------"
    echo " ACTION REQUIRED - Node.js is needed for full-quality"
    echo " YouTube downloads. UltraSinger still works, but YouTube"
    echo " downloads stay limited to 360p until this is done:"
    echo ""
    echo "   1. Install Node.js LTS:"
    echo "        macOS:  brew install node   (or https://nodejs.org)"
    echo "        Debian/Ubuntu:  sudo apt install nodejs npm"
    echo "        Fedora:         sudo dnf install nodejs"
    echo "        or download from https://nodejs.org"
    echo "   2. Open a NEW terminal (so Node.js is on the PATH)"
    echo "   3. Run the installer again:"
    echo "        $RERUN"
    echo "------------------------------------------------------------"
    echo ""
    exit 2
fi

if ! command -v git >/dev/null 2>&1; then
    echo ""
    echo "------------------------------------------------------------"
    echo " git was not found, so the provider source cannot be fetched."
    echo " Install git, open a NEW terminal, and run the installer"
    echo " again:  $RERUN"
    echo "------------------------------------------------------------"
    echo ""
    exit 2
fi

# --- Fetch and build the provider -------------------------------------------
mkdir -p .potoken
if [ -d "$PROVIDER_DIR/.git" ]; then
    echo "Updating provider source..."
    git -C "$PROVIDER_DIR" pull --ff-only >/dev/null 2>&1
else
    echo "Downloading provider source..."
    git clone --depth 1 "$PROVIDER_REPO" "$PROVIDER_DIR" >/dev/null 2>&1
fi
if [ ! -d "$PROVIDER_DIR/server" ]; then
    echo "Could not download the provider source. Check your internet"
    echo "connection and run $RERUN again."
    echo "If you are behind a corporate proxy: git and npm honor the same"
    echo "HTTP_PROXY/HTTPS_PROXY/NO_PROXY variables as the rest of this"
    echo "installer - set them (and UV_NATIVE_TLS=1 for TLS-inspecting"
    echo "proxies) and re-run."
    exit 3
fi

echo "Building provider (npm install + tsc, this can take a minute)..."
( cd "$PROVIDER_DIR/server" && npm install --no-audit --no-fund >/dev/null 2>&1 && npx --yes tsc >/dev/null 2>&1 )

if [ -f "$SERVER_ENTRY" ]; then
    echo "Done. The GUI starts the provider automatically on launch -"
    echo "full-quality YouTube downloads are enabled."
    exit 0
else
    echo "The provider build did not complete. Run $RERUN again;"
    echo "if it keeps failing, YouTube downloads stay limited to 360p."
    echo "If you are behind a corporate proxy: git and npm honor the same"
    echo "HTTP_PROXY/HTTPS_PROXY/NO_PROXY variables as the rest of this"
    echo "installer - set them (and UV_NATIVE_TLS=1 for TLS-inspecting"
    echo "proxies) and re-run."
    exit 3
fi
