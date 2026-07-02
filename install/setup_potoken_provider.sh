#!/bin/bash
# Set up the bgutil PO-token provider (Node.js server) so YouTube downloads
# get full-quality formats. Non-fatal: if Node.js is missing the rest of the
# install still succeeds and the GUI shows a setup hint instead.
#
# Must be run from the repository root (the install scripts cd there first).

set +e  # never abort the parent install script

PROVIDER_DIR=".potoken/bgutil-ytdlp-pot-provider"
PROVIDER_REPO="https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git"
SERVER_ENTRY="$PROVIDER_DIR/server/build/main.js"

echo ""
echo "Setting up the PO-token provider (full-quality YouTube downloads)..."

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo "  Node.js not found - skipping. Install Node.js from https://nodejs.org"
    echo "  and re-run this script to enable full-quality YouTube downloads."
    exit 0
fi
if ! command -v git >/dev/null 2>&1; then
    echo "  git not found - skipping PO-token provider setup."
    exit 0
fi

mkdir -p .potoken
if [ -d "$PROVIDER_DIR/.git" ]; then
    echo "  Updating provider source..."
    git -C "$PROVIDER_DIR" pull --ff-only >/dev/null 2>&1
else
    echo "  Cloning provider source..."
    git clone --depth 1 "$PROVIDER_REPO" "$PROVIDER_DIR" >/dev/null 2>&1
fi
if [ ! -d "$PROVIDER_DIR/server" ]; then
    echo "  Could not obtain the provider source - skipping."
    exit 0
fi

echo "  Building provider (npm install + tsc)..."
( cd "$PROVIDER_DIR/server" && npm install --no-audit --no-fund >/dev/null 2>&1 && npx --yes tsc >/dev/null 2>&1 )

if [ -f "$SERVER_ENTRY" ]; then
    echo "  PO-token provider ready. The GUI starts it automatically on launch."
else
    echo "  PO-token provider build did not complete - the GUI will show a hint."
fi
exit 0
