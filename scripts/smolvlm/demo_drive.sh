#!/bin/bash
# Driving hazard detection: start llama-server, send request, stop server
set -e

LLAMA_DIR="/home/bokai/capstone/third-party/llama.cpp/build-vulkan/llama-b9496"
export LD_LIBRARY_PATH="$LLAMA_DIR:$LD_LIBRARY_PATH"

MODEL_DIR="/home/bokai/capstone/models/SmolVLM2-2.2B-Instruct-GGUF"
MODEL="$MODEL_DIR/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"
MMPROJ="$MODEL_DIR/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf"

IMAGE="${1:-/home/bokai/capstone/scripts/smolvlm/smolvlm-demo.png}"
PORT=8081

cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

echo "=== Starting llama-server (Vulkan) ==="
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
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "Server ready on port $PORT"
        break
    fi
    if [ "$i" -eq 30 ]; then echo "Server failed to start"; exit 1; fi
    sleep 1
done

echo "=== Sending request ==="
export IMAGE PORT

python3 << 'PYEOF'
import json, base64, os, urllib.request

image_path = os.environ['IMAGE']
with open(image_path, 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

media_type = 'image/jpeg' if image_path.lower().endswith(('.jpg', '.jpeg')) else 'image/png'

prompt = (
    "[Dashcam front view] I am driving. "
    "Describe what you see ahead. "
    "Is there any danger? "
    "Answer in 1-2 sentences. "
    "Start with: Danger: yes/no. Then explain briefly."
)

payload = {
    'model': 'smolvlm',
    'messages': [{
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:{media_type};base64,{b64}'}},
            {'type': 'text', 'text': prompt}
        ]
    }],
    'max_tokens': 64,
    'temperature': 0.01
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