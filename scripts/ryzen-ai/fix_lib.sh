#!/usr/bin/env bash
# Fix RyzenAI library symlink and LD_LIBRARY_PATH.
# Run this BEFORE any inference: source scripts/fix_lib.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv-ryzen-ai"

# Find the deployment lib directory
if [ -d "$VENV_DIR/deployment/lib" ]; then
    LIB_DIR="$VENV_DIR/deployment/lib"
elif [ -d "$VENV_DIR/lib/python3.12/site-packages/onnxruntime/capi" ]; then
    LIB_DIR="$VENV_DIR/lib/python3.12/site-packages/onnxruntime/capi"
else
    echo "ERROR: Cannot find RyzenAI library directory"
    return 1 2>/dev/null || exit 1
fi

# Create symlink in project root if missing
TARGET="$LIB_DIR/libonnxruntime_providers_ryzenai.so"
SYMLINK="$PROJECT_DIR/libonnxruntime_providers_ryzenai.so"
if [ ! -L "$SYMLINK" ] && [ ! -f "$SYMLINK" ]; then
    ln -s "$TARGET" "$SYMLINK"
    echo "[fix_lib] Created symlink: $SYMLINK -> $TARGET"
else
    echo "[fix_lib] Symlink already exists: $SYMLINK"
fi

# Prepend deployment lib to LD_LIBRARY_PATH
if [[ ":$LD_LIBRARY_PATH:" != *":$LIB_DIR:"* ]]; then
    export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"
    echo "[fix_lib] Added $LIB_DIR to LD_LIBRARY_PATH"
fi

# Make it discoverable for child processes only, not persistent in env
export RYZEN_AI_LIB_FIXED=1
echo "[fix_lib] Done."