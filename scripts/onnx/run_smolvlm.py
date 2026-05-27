#!/usr/bin/env python3
"""
SmolVLM-Instruct ONNX inference with hybrid execution:
  Vision encoder → subprocess (ROCm / OpenVINO / CPU)
  Decoder       → local process (CPU / NPU via OGA)

Usage (single env - OpenVINO or CPU):
  python scripts/run_smolvlm.py -i image.jpg

Usage (with ROCm in separate venv):
  python scripts/run_smolvlm.py -i image.jpg \
    --vision-python /path/to/rocm-venv/bin/python \
    --vision-provider rocm
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from transformers import AutoProcessor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models", "SmolVLM-Instruct-onnx")
ONNX_DIR = os.path.join(MODEL_DIR, "onnx")


def pick_onnx(prefix):
    for variant in ["q4f16", "fp16", "int8", "bnb4", ""]:
        path = os.path.join(ONNX_DIR, f"{prefix}_{variant}.onnx") if variant else os.path.join(ONNX_DIR, f"{prefix}.onnx")
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="SmolVLM-Instruct ONNX inference (hybrid)")
    parser.add_argument("-i", "--images", type=str, nargs="+", required=True)
    parser.add_argument("-p", "--prompt", type=str, default="Can you describe this image?")
    parser.add_argument("-m", "--model-dir", type=str, default=MODEL_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vision-python", type=str, default=None,
                        help="Python executable for vision encoder (e.g. ROCm venv python)")
    parser.add_argument("--vision-provider", type=str, default="auto",
                        choices=["auto", "rocm", "openvino", "cpu"])
    parser.add_argument("--timings", "-t", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir
    onnx_dir = os.path.join(model_dir, "onnx")

    # Load config
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    text_cfg = config["text_config"]
    IMAGE_TOKEN_ID = config["image_token_id"]
    EOS_TOKEN_ID = text_cfg.get("eos_token_id", 0)
    NUM_LAYERS = text_cfg["num_hidden_layers"]
    NUM_KV_HEADS = text_cfg.get("num_key_value_heads", text_cfg["num_attention_heads"])
    HEAD_DIM = text_cfg.get("head_dim", 64)

    decoder_path = pick_onnx("decoder_model_merged")
    embed_path = pick_onnx("embed_tokens")
    if not decoder_path or not embed_path:
        print("[ERROR] Decoder/embed ONNX not found. Run: bash scripts/download_smolvlm.sh")
        sys.exit(1)

    # ── Step 1: Run vision encoder (subprocess) ──
    import onnxruntime as _ort
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        features_path = f.name

    vision_python = args.vision_python or sys.executable
    vision_script = os.path.join(ROOT_DIR, "scripts", "vision_encoder.py")
    vision_cmd = [
        vision_python, vision_script,
        "--image", *args.images,
        "--prompt", args.prompt,
        "--output", features_path,
        "--provider", args.vision_provider,
    ]
    if args.timings:
        vision_cmd.append("--timings")

    result = subprocess.run(vision_cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] Vision encoder failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            if "[vision_encoder]" in line:
                print(f"  {line}")

    image_features = np.load(features_path)
    os.unlink(features_path)

    # ── Step 2: Decode on CPU ──
    import onnxruntime
    cpu_opts = onnxruntime.SessionOptions()
    cpu_opts.enable_mem_pattern = False

    embed_session = onnxruntime.InferenceSession(embed_path, cpu_opts, providers=["CPUExecutionProvider"])
    decoder_session = onnxruntime.InferenceSession(decoder_path, cpu_opts, providers=["CPUExecutionProvider"])

    processor = AutoProcessor.from_pretrained(model_dir)

    images = args.images
    messages = [{
        "role": "user",
        "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": args.prompt}],
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, return_tensors="np")

    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    batch_size = 1

    # KV cache (float16)
    past_key_values = {}
    for layer in range(NUM_LAYERS):
        for kv in ("key", "value"):
            past_key_values[f"past_key_values.{layer}.{kv}"] = np.zeros(
                [batch_size, NUM_KV_HEADS, 0, HEAD_DIM], dtype=np.float16
            )

    generated_tokens = []
    rng = np.random.default_rng(args.seed)

    print(f"\n  Decoder: generating up to {args.max_new_tokens} tokens...")
    print("  Output: ", end="", flush=True)

    for step in range(args.max_new_tokens):
        inputs_embeds = embed_session.run(None, {"input_ids": input_ids})[0]

        # Merge vision features at <image> positions (first step only)
        if step == 0:
            image_positions = np.where(input_ids[0] == IMAGE_TOKEN_ID)[0]
            if len(image_positions) > 0:
                n_pos = len(image_positions)
                n_img = image_features.shape[0]
                for i in range(min(n_img, n_pos)):
                    inputs_embeds[0, image_positions[i]] = image_features[i, 0]
                # If more features than positions, append them
                if n_img > n_pos:
                    extra = image_features[n_pos:, 0]
                    inputs_embeds = np.concatenate([inputs_embeds, extra.reshape(1, -1, 2048)], axis=1)
                    attention_mask = np.concatenate([
                        attention_mask,
                        np.ones((batch_size, extra.shape[0]), dtype=np.int64)
                    ], axis=1)

        decoder_inputs = dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **past_key_values)
        logits, *present_kv = decoder_session.run(None, decoder_inputs)

        last_logits = logits[0, -1, :]
        if args.temperature > 0 and args.temperature != 1.0:
            last_logits = last_logits / args.temperature
        probs = np.exp(last_logits - np.max(last_logits))
        probs = probs / np.sum(probs)
        next_token = np.array([[rng.choice(len(probs), p=probs)]])

        token_id = next_token[0, 0]
        generated_tokens.append(token_id)
        print(processor.decode([token_id]), end="", flush=True)

        if token_id == EOS_TOKEN_ID:
            print()
            break

        input_ids = next_token.astype(np.int64)
        attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
        for j, key in enumerate(past_key_values):
            past_key_values[key] = present_kv[j]

    decoded = processor.decode(generated_tokens, skip_special_tokens=True)
    print(f"\n\n  Full output:\n  {decoded}")


if __name__ == "__main__":
    main()