import base64
import json
import os
import subprocess
import threading
import time
import urllib.request

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__, template_folder="templates")

LLAMA_DIR = "/home/bokai/capstone/third-party/llama.cpp/build-vulkan/llama-b9496"
MODEL = "/home/bokai/capstone/models/SmolVLM2-2.2B-Instruct-GGUF/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"
MMPROJ = "/home/bokai/capstone/models/SmolVLM2-2.2B-Instruct-GGUF/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf"
LLAMA_PORT = 8081
server_proc = None
server_lock = threading.Lock()


def ensure_server():
    global server_proc
    with server_lock:
        if server_proc is not None:
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}/health")
                urllib.request.urlopen(req, timeout=2)
                return
            except Exception:
                server_proc.kill()
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{LLAMA_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
        server_proc = subprocess.Popen(
            [f"{LLAMA_DIR}/llama-server", "-m", MODEL, "--mmproj", MMPROJ,
             "--port", str(LLAMA_PORT), "-ngl", "99", "--mmproj-offload", "--no-warmup"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
        )
        for i in range(30):
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}/health")
                urllib.request.urlopen(req, timeout=2)
                return
            except Exception:
                if i == 29:
                    raise RuntimeError("llama-server failed to start")
                time.sleep(1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def serve_video():
    video_path = os.path.join(os.path.dirname(__file__), "smolvlm-demo.mp4")
    return send_file(video_path, mimetype="video/mp4")


@app.route("/analyze", methods=["POST"])
def analyze():
    ensure_server()
    data = request.get_json()
    image_b64 = data["image"]
    prompt = data.get("prompt", (
        "[Dashcam front view] I am driving. "
        "Describe what you see ahead. "
        "Is there any danger? "
        "Answer in 1-2 sentences. "
        "Start with: Danger: yes/no. Then explain briefly."
    ))

    payload = {
        "model": "smolvlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ]
        }],
        "max_tokens": 64,
        "temperature": 0.01,
    }
    req_data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions",
        data=req_data,
        headers={"Content-Type": "application/json"},
    )

    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    content = result["choices"][0]["message"]["content"]
    return jsonify({"text": content})


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import atexit
    atexit.register(lambda: server_proc.kill() if server_proc else None)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
