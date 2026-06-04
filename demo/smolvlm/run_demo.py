import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def extract_keyframes(video_path, output_dir, fps=2.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total / vid_fps if vid_fps > 0 else 0
    frame_interval = max(1, int(vid_fps / fps))
    indices = [i for i in range(0, total, frame_interval)]
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            ts = idx / fps if fps > 0 else 0
            p = frames_dir / f"frame_{idx:04d}_t{ts:.1f}s.png"
            cv2.imwrite(str(p), frame)
            paths.append(p)
    cap.release()
    print(f"Extracted {len(paths)} keyframes from {total} frames ({duration:.1f}s)")
    return sorted(paths)


def query_llama(image_path, port, prompt_template, max_tokens=64, temperature=0.01):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = str(image_path).lower()
    media_type = "image/jpeg" if ext.endswith((".jpg", ".jpeg")) else "image/png"

    payload = {
        "model": "smolvlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                {"type": "text", "text": prompt_template},
            ]
        }],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    return content, usage


def create_output_collage(results, output_path):
    images = []
    draw_texts = []
    for img_path, text, _ in results:
        img = Image.open(img_path).convert("RGB")
        max_w = 640
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
        images.append(img)
        draw_texts.append(text)

    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    pad = 160
    total_h = sum(max(img.height, cell_h) for img in images) + pad * len(images) + 20

    collage = Image.new("RGB", (cell_w + 40, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(collage)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    y = 10
    for img, text in zip(images, draw_texts):
        collage.paste(img, (20, y))
        y += img.height + 8
        for line in text.split("\n"):
            for wrapped_line in [line[i:i+80] for i in range(0, len(line), 80)]:
                draw.text((20, y), wrapped_line, fill=(200, 200, 200), font=font)
                y += 20
        y += 16

    collage.save(str(output_path))
    print(f"Collage saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SmolVLM Driving Hazard Detection Demo")
    parser.add_argument("--video", default="smolvlm-demo.mp4",
                        help="Input driving video path")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frames per second to extract (default: 2.0)")
    parser.add_argument("--port", type=int, default=8081,
                        help="llama-server port")
    parser.add_argument("--llama-dir",
                        default="/home/bokai/capstone/third-party/llama.cpp/build-vulkan/llama-b9496",
                        help="llama.cpp build directory")
    parser.add_argument("--model",
                        default="/home/bokai/capstone/models/SmolVLM2-2.2B-Instruct-GGUF/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
                        help="GGUF model path")
    parser.add_argument("--mmproj",
                        default="/home/bokai/capstone/models/SmolVLM2-2.2B-Instruct-GGUF/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf",
                        help="MMProj path")
    parser.add_argument("--no-server", action="store_true",
                        help="Skip starting server (use existing)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup requests")
    args = parser.parse_args()

    llama_server = Path(args.llama_dir) / "llama-server"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = args.video
    if not Path(video_path).exists():
        video_path = str(Path(__file__).parent / args.video)
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    extract_keyframes(video_path, out_dir, args.fps)

    if not args.no_server:
        print("\n=== Starting llama-server (Vulkan) ===")
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{args.llama_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        server_proc = subprocess.Popen(
            [str(llama_server), "-m", args.model, "--mmproj", args.mmproj,
             "--port", str(args.port), "-ngl", "99", "--mmproj-offload", "--no-warmup"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
        )

        for i in range(30):
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{args.port}/health")
                urllib.request.urlopen(req, timeout=2)
                print(f"Server ready on port {args.port}")
                break
            except Exception:
                if i == 29:
                    print("Server failed to start")
                    server_proc.kill()
                    sys.exit(1)
                time.sleep(1)

    prompt = (
        "[Dashcam front view] I am driving. "
        "Describe what you see ahead. "
        "Is there any danger? "
        "Answer in 1-2 sentences. "
        "Start with: Danger: yes/no. Then explain briefly."
    )

    if args.warmup > 0 and not args.no_server:
        print(f"\n=== Warmup ({args.warmup}x) ===")
        warmup_img = str(out_dir / "frames" / sorted(os.listdir(out_dir / "frames"))[0])
        for i in range(args.warmup):
            try:
                text, _ = query_llama(warmup_img, args.port, prompt, max_tokens=16, temperature=0.01)
                print(f"  Warmup {i+1}/{args.warmup}: {text[:50]}...")
            except Exception as e:
                print(f"  Warmup {i+1}/{args.warmup}: {e}")

    frame_paths = sorted(Path(out_dir / "frames").glob("*.png"))
    print(f"\n=== Processing {len(frame_paths)} frames ===")
    results = []
    for i, fp in enumerate(frame_paths):
        print(f"  [{i+1}/{len(frame_paths)}] {fp.name} ... ", end="", flush=True)
        try:
            text, usage = query_llama(str(fp), args.port, prompt)
            tokens = usage.get("completion_tokens", 0)
            print(f"Danger assessment ({tokens} tokens): {text[:80]}")
            results.append((fp, text, usage))
        except Exception as e:
            print(f"Error: {e}")
            results.append((fp, f"[Error: {e}]", {}))

    output = {"video": str(video_path), "num_frames": len(results), "results": []}
    for fp, text, usage in results:
        output["results"].append({
            "frame": fp.name, "timestamp": fp.stem.split("_t")[-1].replace("s", "") if "_t" in fp.stem else "",
            "hazard_text": text, "tokens": usage.get("completion_tokens", 0),
        })
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {json_path}")

    create_output_collage(results, out_dir / "demo_collage.png")

    print("\n=== Summary ===")
    for r in output["results"]:
        print(f"  {r['frame']}: {r['hazard_text'][:100]}")

    if not args.no_server:
        server_proc.kill()
        print("Server stopped.")


if __name__ == "__main__":
    main()