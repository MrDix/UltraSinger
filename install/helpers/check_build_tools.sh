#!/bin/bash
# Check for the tools that are needed to build the dependencies without
# pre-built packages for Linux/macOS (e.g. diffq of the vocal separation and
# pyhyphen): a C compiler and the headers of the Python that uv will use.
# Without them the install fails late, deep in a package build, with a cryptic
# compiler error. Exit codes: 0 = all present, 1 = something is missing (the
# command that installs it has been printed).
#
# Must be run from the repository root, with uv on the PATH.

if [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
    # macOS: the compiler comes with the command line developer tools; every
    # Python (python.org, Homebrew, uv) ships its headers.
    if xcode-select -p >/dev/null 2>&1; then
        exit 0
    fi
    echo ""
    echo "------------------------------------------------------------"
    echo " ACTION REQUIRED - some dependencies are built from source and"
    echo " need the command line developer tools (C compiler). Install them with:"
    echo "     xcode-select --install"
    echo " and run the installer again."
    echo "------------------------------------------------------------"
    echo ""
    exit 1
fi

MISSING_CC=""
if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
    MISSING_CC=1
fi

# The interpreter uv will create the environment with (a fresh one: --system
# skips the existing .venv). A uv-managed Python brings its headers, a system
# Python needs its development package.
MISSING_HEADERS=""
PY_VERSION=""
PY="$(uv python find --system '>=3.12,<3.14' 2>/dev/null)"
if [ -n "$PY" ]; then
    PY_VERSION="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
    if ! "$PY" -c 'import os, sys, sysconfig; sys.exit(0 if os.path.isfile(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)' >/dev/null 2>&1; then
        MISSING_HEADERS=1
    fi
fi

if [ -z "$MISSING_CC" ] && [ -z "$MISSING_HEADERS" ]; then
    exit 0
fi

PY_PKG="python${PY_VERSION:-3}"
echo ""
echo "------------------------------------------------------------"
echo " ACTION REQUIRED - some dependencies are built from source and"
if [ -n "$MISSING_CC" ] && [ -n "$MISSING_HEADERS" ]; then
    echo " need a C compiler and the Python $PY_VERSION headers ($PY)."
elif [ -n "$MISSING_CC" ]; then
    echo " need a C compiler."
else
    echo " need the Python $PY_VERSION headers ($PY)."
fi
echo " Install them, e.g. with:"
echo "     Debian/Ubuntu:  sudo apt install build-essential ${PY_PKG}-dev"
echo "     Fedora:         sudo dnf install gcc ${PY_PKG}-devel"
echo "     Arch:           sudo pacman -S base-devel"
echo " and run the installer again."
echo "------------------------------------------------------------"
echo ""
exit 1
