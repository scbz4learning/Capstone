#!/bin/bash
set -e

LLAMA_DIR="/home/bokai/capstone/third-party/llama.cpp/build-vulkan/llama-b9496"
MODEL_DIR="/home/bokai/capstone/models/SmolVLM2-2.2B-Instruct-GGUF"
MODEL="$MODEL_DIR/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"
MMPROJ="$MODEL_DIR/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf"
LLAMA_PORT=8081
APP_PORT=5000
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="/home/bokai/capstone/.venv/bin/python"

export LD_LIBRARY_PATH="$LLAMA_DIR:$LD_LIBRARY_PATH"

if pgrep -x llama-server > /dev/null 2>&1; then
    kill $(pgrep -x llama-server) 2>/dev/null
    sleep 1
fi

echo "[1/2] Starting llama-server (Vulkan, 2.2B Q4_K_M)..."
"$LLAMA_DIR/llama-server" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --port "$LLAMA_PORT" \
    -ngl 99 \
    --mmproj-offload \
    --no-warmup \
    > /tmp/llama-server.log 2>&1 &
LLAMA_PID=$!

for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$LLAMA_PORT/health" > /dev/null; then
        echo "llama-server ready on port $LLAMA_PORT"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "llama-server failed to start"
        cat /tmp/llama-server.log
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== Warmup (3x) ==="
WARMUP_IMG="/home/bokai/capstone/scripts/smolvlm/smolvlm-demo.png"
if [ -f "$WARMUP_IMG" ]; then
    for i in 1 2 3; do
        $VENV_PYTHON -c "
import base64, json, urllib.request
with open('$WARMUP_IMG', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()
payload = {
    'model': 'smolvlm',
    'messages': [{'role':'user','content': [
        {'type':'image_url','image_url':{'url':f'data:image/png;base64,{b64}'}},
        {'type':'text','text':'[Dashcam front view] I am driving. Describe what you see ahead. Is there any danger? Answer in 1-2 sentences. Start with: Danger: yes/no. Then explain briefly.'}
    ]}],
    'max_tokens': 64, 'temperature': 0.01,
}
data = json.dumps(payload).encode()
req = urllib.request.Request('http://127.0.0.1:$LLAMA_PORT/v1/chat/completions', data=data, headers={'Content-Type':'application/json'})
result = json.loads(urllib.request.urlopen(req, timeout=60).read())
print(f'  Warmup $i/3: {result[\"choices\"][0][\"message\"][\"content\"][:40]}')
" || echo "  Warmup $i/3: skipped"
    done
else
    echo "  No warmup image found, skipping"
fi

echo ""
echo "[2/2] Starting SmolVLM Hazard Detection on port $APP_PORT..."
cd "$APP_DIR" && $VENV_PYTHON app.py > /tmp/smolvlm-app.log 2>&1 &
SMOLVLM_PID=$!

trap 'kill $LLAMA_PID 2>/dev/null; kill $SMOLVLM_PID 2>/dev/null' EXIT

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SmolVLM Hazard Detection Demo"
echo "  http://localhost:$APP_PORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Press Ctrl+C to stop"

wait $SMOLVLM_PID
kill $LLAMA_PID 2>/dev/null