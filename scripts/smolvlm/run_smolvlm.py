import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


def main():
    parser = argparse.ArgumentParser(
        description="Run SmolVLM inference: describe images or answer questions about them."
    )
    parser.add_argument("--images", nargs="+", default=[],
                        help="Paths to input images")
    parser.add_argument("--prompt", default="Describe the images briefly.",
                        help="Text prompt (default: 'Describe the images briefly.')")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct",
                        help="Model ID (default: HuggingFaceTB/SmolVLM-Instruct)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Compute device (default: cuda)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"],
                        help="Model dtype (default: bfloat16)")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                        help="Attention implementation (default: sdpa)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum new tokens to generate (default: 128)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (default: None = greedy)")
    parser.add_argument("--output-dir", default="smolvlm_output",
                        help="Directory for saving output (default: smolvlm_output)")
    parser.add_argument("--stream", action="store_true",
                        help="Stream tokens as they are generated")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

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

    print(f"Loading SmolVLM model: {args.model}")
    print(f"  device={args.device}, dtype={args.dtype}, attn={args.attn}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn
    ).to(device).eval()

    prompt_content = [{"type": "image"} for _ in range(num_images)]
    prompt_content.append({"type": "text", "text": args.prompt})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": prompt_content}],
        add_generation_prompt=True,
    )
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
              for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_tokens,
        min_new_tokens=min(16, args.max_tokens),
        use_cache=True,
        do_sample=args.temperature is not None,
    )
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature

    print(f"\nGenerating (max {args.max_tokens} tokens)...")
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()

    if args.stream:
        from transformers import TextIteratorStreamer
        import threading
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True)
        gen_kwargs["streamer"] = streamer
        thread = threading.Thread(target=model.generate, kwargs={**inputs, **gen_kwargs})
        thread.start()
        print("\nOutput: ", end="", flush=True)
        output_text = ""
        for token in streamer:
            print(token, end="", flush=True)
            output_text += token
        thread.join()
        print()
    else:
        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)
        output_text = processor.decode(generated[0], skip_special_tokens=True)
        prompt_len = len(processor.tokenizer.encode(prompt, add_special_tokens=False))
        output_text = output_text.split(prompt)[-1].strip() if prompt in output_text else output_text
        print(f"\nOutput:\n{output_text}")

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - t0

    ttft = 0.0
    tpot = 0.0
    if args.stream and hasattr(streamer, "_timings"):
        ttft = streamer._timings.get("ttft", 0)
        tpot = streamer._timings.get("tpot", 0)

    result = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "prompt": args.prompt,
        "num_images": num_images,
        "output": output_text,
        "num_tokens": len(processor.tokenizer.encode(output_text, add_special_tokens=False)),
        "total_time_s": round(elapsed, 3),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n--- Summary ---")
    print(f"Total time:  {elapsed:.3f}s")
    print(f"Output saved to {out_dir.resolve()}/result.json")


if __name__ == "__main__":
    main()