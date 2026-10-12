#!/bin/bash
# Set up the bgutil PO-token provider (Node.js server) so video downloads
# get full-quality formats. Non-fatal: the parent install script keeps going
# regardless. Exit codes: 0 = provider ready, 2 = Node.js/git missing or
# Node.js too old, 3 = download or build failed. Arg $1 = the install script
# to name in the re-run hint.
#
# Must be run from the repository root (the install scripts cd there first).

set +e  # never abort the parent install script

RERUN="${1:-the install script you just ran}"
PROVIDER_DIR=".potoken/bgutil-ytdlp-pot-provider"
PROVIDER_REPO="https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git"
# The provider release that is built, pinned by its tag and verified by its
# commit, so neither a newer release nor a moved tag runs unreviewed code.
# Keep it at the version of the bgutil-ytdlp-pot-provider plugin in uv.lock
# (server and plugin are released together) and in sync with
# setup_potoken_provider.bat. NODE_MIN_MAJOR is the release's "engines" entry.
PROVIDER_TAG="2.0.2"
PROVIDER_COMMIT="26475e9d3665b972c4aeb83b4dbaafcf64b88438"
NODE_MIN_MAJOR=22
SERVER_DIR="$PROVIDER_DIR/server"
SERVER_ENTRY="$SERVER_DIR/build/main.js"
# The commit the current build was made from
BUILD_STAMP="$SERVER_DIR/build/.ultrasinger-commit"

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

NODE_MAJOR=""
if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null)"
    case "$NODE_MAJOR" in
        ''|*[!0-9]*) NODE_MAJOR=0 ;;
    esac
fi

if [ -z "$NODE_MAJOR" ] || [ "$NODE_MAJOR" -lt "$NODE_MIN_MAJOR" ]; then
    echo ""
    echo "------------------------------------------------------------"
    if [ -z "$NODE_MAJOR" ]; then
        echo " ACTION REQUIRED - Node.js is needed for full-quality"
        echo " video downloads. UltraSinger still works, but video"
        echo " downloads stay limited to 360p until this is done:"
    else
        echo " ACTION REQUIRED - Node.js $NODE_MIN_MAJOR or newer is needed for"
        echo " full-quality video downloads (found $(node --version 2>/dev/null))."
        echo " The provider is not updated until this is done:"
    fi
    echo ""
    echo "   1. Install Node.js LTS ($NODE_MIN_MAJOR or newer):"
    echo "        macOS:  brew install node   (or https://nodejs.org)"
    echo "        Linux:  https://nodejs.org/en/download (the packages of"
    echo "                many distributions are older than $NODE_MIN_MAJOR)"
    echo "   2. Open a NEW terminal (so Node.js is on the PATH)"
    echo "   3. Run the installer again:"
    echo "        $RERUN"
    echo "------------------------------------------------------------"
    echo ""
    exit 2
fi

# --- Already built from the pinned release? ---------------------------------
# Also needs the installed dependencies (npm ci writes the hidden lockfile):
# security software sometimes removes them, and then the build must be redone.
if [ -f "$SERVER_ENTRY" ] && [ -f "$SERVER_DIR/node_modules/.package-lock.json" ] \
        && [ "$(tr -d '\r\n' 2>/dev/null < "$BUILD_STAMP")" = "$PROVIDER_COMMIT" ]; then
    echo "The provider (release $PROVIDER_TAG) is already built -"
    echo "full-quality video downloads are enabled."
    exit 0
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

# --- Fetch the pinned release -------------------------------------------------
# Every git command on the provider folder checks for its .git first: in a
# folder without one, git would act on the UltraSinger repository instead.
mkdir -p .potoken
if [ -d "$PROVIDER_DIR/.git" ]; then
    echo "Updating the provider source to release $PROVIDER_TAG..."
    git -C "$PROVIDER_DIR" fetch --quiet --depth 1 --force origin \
        "refs/tags/$PROVIDER_TAG:refs/tags/$PROVIDER_TAG" >/dev/null 2>&1
else
    echo "Downloading the provider source (release $PROVIDER_TAG)..."
    # Into a new folder first: a provider folder without .git (e.g. restored
    # from a backup that skips .git) is only replaced once the download worked.
    rm -rf "$PROVIDER_DIR.new"
    git clone --quiet --depth 1 --branch "$PROVIDER_TAG" "$PROVIDER_REPO" "$PROVIDER_DIR.new" >/dev/null 2>&1
    if [ -d "$PROVIDER_DIR.new/.git" ] \
            && [ "$(git -C "$PROVIDER_DIR.new" rev-parse HEAD 2>/dev/null)" = "$PROVIDER_COMMIT" ]; then
        rm -rf "$PROVIDER_DIR" && mv "$PROVIDER_DIR.new" "$PROVIDER_DIR"
    fi
    rm -rf "$PROVIDER_DIR.new"
fi
SRC_COMMIT=""
if [ -d "$PROVIDER_DIR/.git" ]; then
    # Check out exactly the pinned release; this also drops local changes to it
    git -C "$PROVIDER_DIR" -c advice.detachedHead=false checkout --quiet --force \
        "refs/tags/$PROVIDER_TAG" >/dev/null 2>&1
    SRC_COMMIT="$(git -C "$PROVIDER_DIR" rev-parse HEAD 2>/dev/null)"
fi
if [ "$SRC_COMMIT" != "$PROVIDER_COMMIT" ]; then
    echo "Could not download the provider source (release $PROVIDER_TAG)."
    echo "Check your internet connection and run $RERUN again."
    echo "If you are behind a corporate proxy: git and npm honor the same"
    echo "HTTP_PROXY/HTTPS_PROXY/NO_PROXY variables as the rest of this"
    echo "installer - set them (and UV_SYSTEM_CERTS=1 for TLS-inspecting"
    echo "proxies) and re-run."
    exit 3
fi

# --- Build it ------------------------------------------------------------------
echo "Building provider (npm ci + tsc, this can take a minute)..."
# Start from scratch: a build of another release must neither be started by
# the app nor make a failed build look successful (tsc's incremental state
# would skip unchanged files, too). npm ci installs exactly the release's
# package-lock.json, and tsc is the compiler it installed (npx --no never
# downloads one).
rm -rf "$SERVER_DIR/build" "$SERVER_DIR/tsconfig.tsbuildinfo"
( cd "$SERVER_DIR" && npm ci --include=dev --no-audit --no-fund >/dev/null 2>&1 && npx --no -- tsc >/dev/null 2>&1 )
BUILD_RC=$?

if [ "$BUILD_RC" -eq 0 ] && [ -f "$SERVER_ENTRY" ]; then
    printf '%s\n' "$PROVIDER_COMMIT" > "$BUILD_STAMP"
    # The GUI's first-launch bootstrap starts the server itself right after
    # this script, so the warm-up below would only double the wait.
    if [ -n "${ULTRASINGER_POTOKEN_SKIP_WARMUP:-}" ]; then
        echo "Provider built. Warm-up skipped (the app manages the server)."
        exit 0
    fi
    # Warm up the provider now so the FIRST app launch is fast. The first time
    # the freshly built server starts, security software scans its many
    # node_modules files, which on corporate machines can take several minutes
    # - during which the app would look broken. Trigger that scan here, where
    # the user expects the installer to work, so later starts are instant.
    echo "Warming up the provider so the first app launch is fast..."
    echo "(On managed/corporate machines antivirus scans the new files on the"
    echo " first start - this can take several minutes. You can leave it running.)"
    node "$SERVER_ENTRY" >/dev/null 2>&1 &
    WARM_PID=$!
    warm_ready=0
    warm_i=0
    while [ "$warm_i" -lt 450 ]; do   # up to ~15 minutes (450 x 2s)
        if node -e "require('http').get('http://127.0.0.1:4416/ping',function(r){process.exit(r.statusCode===200?0:1)}).on('error',function(){process.exit(1)})" >/dev/null 2>&1; then
            warm_ready=1
            break
        fi
        warm_i=$((warm_i + 1))
        sleep 2
    done
    kill "$WARM_PID" >/dev/null 2>&1 || true
    if [ "$warm_ready" = 1 ]; then
        echo "Provider warmed up - the GUI will start it quickly from now on."
    else
        echo "Provider warm-up did not finish in time; the first app launch may"
        echo "still take a few minutes while security scanning completes."
    fi
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
