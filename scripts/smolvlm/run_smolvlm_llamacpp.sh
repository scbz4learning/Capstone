#!/bin/bash
# Run SmolVLM via llama-server (Vulkan). Start server, send one request, stop.
set -e

LLAMA_DIR="/home/bokai/capstone/third-party/llama.cpp/build-vulkan/llama-b9496"
export LD_LIBRARY_PATH="$LLAMA_DIR:$LD_LIBRARY_PATH"

MODEL_DIR="/home/bokai/capstone/models/SmolVLM2-2.2B-Instruct-GGUF"
MODEL="$MODEL_DIR/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"
MMPROJ="$MODEL_DIR/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf"

PORT=8081
PROMPT="Describe the image briefly."
IMAGE=""
MAX_TOKENS=128
TEMPERATURE=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --prompt|-p) PROMPT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        -n|--max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --temp|--temperature) TEMPERATURE="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -z "$IMAGE" ]; then
    echo "Usage: $0 --image <path> [--prompt <text>] [--port <num>] [-n <tokens>] [--temp <float>]"
    exit 1
fi

cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

"$LLAMA_DIR/llama-server" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --port "$PORT" \
    -ngl 99 \
    --mmproj-offload \
    --no-warmup \
    > /dev/null 2>&1 &
SERVER_PID=$!

for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
    if [ "$i" -eq 30 ]; then echo "Server failed to start"; exit 1; fi
    sleep 1
done

export IMAGE PROMPT MAX_TOKENS TEMPERATURE PORT

python3 << 'PYEOF'
import json, base64, os, urllib.request

image_path = os.environ['IMAGE']
with open(image_path, 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

if image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
    media_type = 'image/jpeg'
elif image_path.lower().endswith('.png'):
    media_type = 'image/png'
else:
    media_type = 'image/png'

payload = {
    'model': 'smolvlm',
    'messages': [{
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:{media_type};base64,{b64}'}},
            {'type': 'text', 'text': os.environ['PROMPT']}
        ]
    }],
    'max_tokens': int(os.environ['MAX_TOKENS']),
    'temperature': float(os.environ['TEMPERATURE'])
}

data = json.dumps(payload).encode()
req = urllib.request.Request(
    f'http://127.0.0.1:{os.environ["PORT"]}/v1/chat/completions',
    data=data,
    headers={'Content-Type': 'application/json'}
)
resp = urllib.request.urlopen(req)
result = json.loads(resp.read())
print(json.dumps(result, indent=2, ensure_ascii=False))
PYEOF