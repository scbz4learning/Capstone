#!/bin/bash
set -e

APP_PORT=5001
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="/home/bokai/capstone/.venv/bin/python"

echo "Starting VGGT 3D Reconstruction on port $APP_PORT..."
cd "$APP_DIR" && $VENV_PYTHON app.py > /tmp/vggt-app.log 2>&1 &
VGGT_PID=$!

trap 'kill $VGGT_PID 2>/dev/null' EXIT

sleep 1
if ! curl -sf "http://127.0.0.1:$APP_PORT/health" > /dev/null; then
    echo "VGGT app failed to start"
    cat /tmp/vggt-app.log
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  VGGT 3D Reconstruction Demo"
echo "  http://localhost:$APP_PORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Press Ctrl+C to stop"

wait $VGGT_PID