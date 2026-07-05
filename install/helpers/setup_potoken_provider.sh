#!/bin/bash
# Set up the bgutil PO-token provider (Node.js server) so video downloads
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
echo " PO-token provider setup (full-quality video downloads)"
echo "============================================================"

# --- Ensure Node.js / npm are available -------------------------------------
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    # Best-effort auto-install where a package manager is clearly available.
    # Transparency: announce the third-party install and, when interactive,
    # ask first (mirrors the winget consent prompt on Windows).
    if command -v brew >/dev/null 2>&1; then
        echo "Node.js not found. It can be installed automatically via Homebrew"
        echo "(MIT-licensed)."
        DO_NODE_INSTALL=1
        if [ -t 0 ]; then
            printf "Install Node.js via brew now? [Y/n]: "
            read -r REPLY || REPLY=""
            case "$REPLY" in
                [Nn]*) DO_NODE_INSTALL="" ;;
            esac
        else
            echo "Non-interactive session - installing automatically."
        fi
        if [ -n "$DO_NODE_INSTALL" ]; then
            echo "Installing Node.js via Homebrew..."
            brew install node >/dev/null 2>&1
        fi
    fi
fi

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo ""
    echo "------------------------------------------------------------"
    echo " ACTION REQUIRED - Node.js is needed for full-quality"
    echo " video downloads. UltraSinger still works, but video"
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
    # Discard our local proxy patch (re-applied below) so ff-only pulls work
    git -C "$PROVIDER_DIR" checkout -- . >/dev/null 2>&1
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
    echo "installer - set them (and UV_SYSTEM_CERTS=1 for TLS-inspecting"
    echo "proxies) and re-run."
    exit 3
fi

# --- Proxy fix: axios needs proxy:false so the httpsAgent does CONNECT ------
# Upstream bug: getFetch() passes a correct proxy-agent as httpsAgent but
# omits axios' `proxy: false`. With HTTP(S)_PROXY set in the environment,
# axios' own env-proxy mode then wins and sends an absolute-form GET to the
# proxy instead of a CONNECT tunnel - enterprise proxies answer 502 and
# token generation fails. Verified against a local sniffing proxy.
SM_TS="$PROVIDER_DIR/server/src/session_manager.ts"
if [ -f "$SM_TS" ] && ! grep -q "proxy: false" "$SM_TS"; then
    echo "Applying proxy workaround (axios proxy:false)..."
    sed -i.bak 's|httpsAgent: proxySpec.asDispatcher(logger),|httpsAgent: proxySpec.asDispatcher(logger),\n                        proxy: false,|' "$SM_TS" \
        && rm -f "$SM_TS.bak"
fi

echo "Building provider (npm install + tsc, this can take a minute)..."
( cd "$PROVIDER_DIR/server" && npm install --no-audit --no-fund >/dev/null 2>&1 && npx --yes tsc >/dev/null 2>&1 )

if [ -f "$SERVER_ENTRY" ]; then
    echo "Done. The GUI starts the provider automatically on launch -"
    echo "full-quality video downloads are enabled."
    exit 0
else
    echo "The provider build did not complete. Run $RERUN again;"
    echo "if it keeps failing, video downloads stay limited to 360p."
    echo "If you are behind a corporate proxy: git and npm honor the same"
    echo "HTTP_PROXY/HTTPS_PROXY/NO_PROXY variables as the rest of this"
    echo "installer - set them (and UV_SYSTEM_CERTS=1 for TLS-inspecting"
    echo "proxies) and re-run."
    exit 3
fi
