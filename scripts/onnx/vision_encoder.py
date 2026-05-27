#!/usr/bin/env python3
"""
Standalone vision encoder for SmolVLM.
Can run with ROCm, OpenVINO, or CPU. Called as subprocess by run_smolvlm.py.

Usage (direct, for testing):
  python scripts/vision_encoder.py \
    --image path/to/image.jpg \
    --output /tmp/features.npy
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image
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
    parser = argparse.ArgumentParser(description="SmolVLM vision encoder (standalone)")
    parser.add_argument("--image", "-i", type=str, nargs="+", required=True)
    parser.add_argument("--prompt", "-p", type=str, default="x")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output .npy file path for image_features")
    parser.add_argument("--model-dir", "-m", type=str, default=MODEL_DIR)
    parser.add_argument("--provider", type=str, default="auto",
                        choices=["auto", "rocm", "openvino", "cpu"],
                        help="Execution provider for vision encoder")
    parser.add_argument("--device-id", type=str, default=None,
                        help="OpenVINO device (e.g. GPU, CPU)")
    parser.add_argument("--timings", "-t", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir
    onnx_dir = os.path.join(model_dir, "onnx")

    # Load config
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    IMAGE_TOKEN_ID = config["image_token_id"]

    vision_path = pick_onnx("vision_encoder")
    if not vision_path:
        print("[ERROR] Vision encoder ONNX not found", file=sys.stderr)
        sys.exit(1)

    # Pick provider
    import onnxruntime
    providers = onnxruntime.get_available_providers()
    provider_choice = args.provider

    if provider_choice == "auto":
        if "ROCMExecutionProvider" in providers:
            provider_choice = "rocm"
        elif "OpenVINOExecutionProvider" in providers:
            provider_choice = "openvino"
        else:
            provider_choice = "cpu"

    if provider_choice == "rocm":
        ep = "ROCMExecutionProvider"
    elif provider_choice == "openvino":
        ep = "OpenVINOExecutionProvider"
    else:
        ep = "CPUExecutionProvider"

    opts = onnxruntime.SessionOptions()
    opts.enable_mem_pattern = False
    ep_opts = {}
    if args.device_id:
        ep_opts["device_type"] = args.device_id
    try:
        vision_session = onnxruntime.InferenceSession(
            vision_path, opts, providers=[(ep, ep_opts)] if ep_opts else [ep]
        )
    except Exception as e:
        print(f"[ERROR] Failed to load with {ep}: {e}", file=sys.stderr)
        print(f"[FALLBACK] Trying CPUExecutionProvider", file=sys.stderr)
        vision_session = onnxruntime.InferenceSession(
            vision_path, opts, providers=["CPUExecutionProvider"]
        )
        provider_choice = "cpu"

    # Load processor
    processor = AutoProcessor.from_pretrained(model_dir)

    # Process images
    images = [Image.open(p).convert("RGB") for p in args.image]
    messages = [{
        "role": "user",
        "content": [{"type": "image"} for _ in args.image] + [{"type": "text", "text": args.prompt}],
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="np")

    pixel_values = inputs["pixel_values"].astype(np.float32)
    pixel_attention_mask = inputs.get("pixel_attention_mask")
    if pixel_attention_mask is not None:
        pixel_attention_mask = pixel_attention_mask.astype(np.bool_)

    # Run vision encoder
    if args.timings:
        t0 = time.time()

    vision_inputs = {"pixel_values": pixel_values}
    if pixel_attention_mask is not None:
        vision_inputs["pixel_attention_mask"] = pixel_attention_mask
    image_features = vision_session.run(["image_features"], vision_inputs)[0]

    if args.timings:
        elapsed = time.time() - t0
        print(f"[vision_encoder] Provider: {provider_choice}, Time: {elapsed:.3f}s", file=sys.stderr)

    # Save features
    np.save(args.output, image_features)
    print(f"[vision_encoder] Saved image_features {image_features.shape} -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()