import argparse
import json
import os
import time
from pathlib import Path

from llama_cpp import Llama
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Run SmolVLM inference with llama.cpp: describe images or answer questions about them."
    )
    parser.add_argument("--images", nargs="+", default=[],
                        help="Paths to input images")
    parser.add_argument("--prompt", default="Describe the images briefly.",
                        help="Text prompt (default: 'Describe the images briefly.')")
    parser.add_argument("--model", required=True,
                        help="Path to the SmolVLM GGUF model file")
    parser.add_argument("--mmproj", required=True,
                        help="Path to the multimodal projector GGUF file")
    parser.add_argument("--n-ctx", type=int, default=4096,
                        help="Context size (default: 4096)")
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="Number of layers to offload to GPU (-1 = all, default: -1)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum new tokens to generate (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p sampling (default: 0.95)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", default="smolvlm_output_llamacpp",
                        help="Directory for saving output (default: smolvlm_output_llamacpp)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose llama.cpp logging")
    parser.add_argument("--chat-format", default="smolvlm",
                        help="Chat format for the vision model (default: smolvlm)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    if args.images:
        for p in args.images:
            if os.path.isfile(p):
                images.append(Image.open(p).convert("RGB"))
        print(f"Loaded {len(images)} images from {len(args.images)} paths")
    else:
        print("No images provided; using a blank dummy image.")
        images = [Image.new("RGB", (384, 384), (128, 128, 128))]
    num_images = len(images)

    print(f"Loading SmolVLM GGUF model: {args.model}")
    print(f"  mmproj={args.mmproj}, n_ctx={args.n_ctx}, n_gpu_layers={args.n_gpu_layers}")

    llm = Llama(
        model_path=args.model,
        mmproj=args.mmproj,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
        verbose=args.verbose,
        chat_format=args.chat_format,
    )

    content = []
    for _ in range(num_images):
        content.append({"type": "image_data", "image": None})
    content.append({"type": "text", "text": args.prompt})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    print(f"\nGenerating (max {args.max_tokens} tokens)...")
    t0 = time.perf_counter()

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        images=images,
    )

    elapsed = time.perf_counter() - t0

    output_text = response["choices"][0]["message"]["content"]
    usage = response.get("usage", {})
    num_tokens = usage.get("completion_tokens", 0)

    print(f"\nOutput:\n{output_text}")

    result = {
        "model": args.model,
        "mmproj": args.mmproj,
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "prompt": args.prompt,
        "num_images": num_images,
        "output": output_text,
        "num_tokens": num_tokens,
        "total_time_s": round(elapsed, 3),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n--- Summary ---")
    print(f"Total time:  {elapsed:.3f}s")
    print(f"Tokens:      {num_tokens}")
    print(f"Output saved to {out_dir.resolve()}/result.json")


if __name__ == "__main__":
    main()