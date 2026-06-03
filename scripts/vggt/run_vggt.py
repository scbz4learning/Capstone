import argparse
import json
import os
import time
from pathlib import Path

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def main():
    parser = argparse.ArgumentParser(
        description="Run VGGT inference on images to produce 3D outputs (point maps, depth, camera poses)."
    )
    parser.add_argument("--images", nargs="+", default=None,
                        help="Paths to input images. If omitted, uses dummy data.")
    parser.add_argument("--model", default="facebook/VGGT-1B",
                        help="Model ID or local path (default: facebook/VGGT-1B)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Compute device (default: cuda)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"],
                        help="Model dtype (default: bfloat16)")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                        help="Attention implementation (default: sdpa)")
    parser.add_argument("--image-size", type=int, default=518,
                        help="Input image size (default: 518)")
    parser.add_argument("--num-images", type=int, default=2,
                        help="Number of dummy images when --images not provided")
    parser.add_argument("--output-dir", default="vggt_output",
                        help="Directory for saving outputs (default: vggt_output)")
    parser.add_argument("--save-vis", action="store_true",
                        help="Save depth visualization as PNG")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    fused_attn = args.attn == "sdpa"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading VGGT model: {args.model}")
    print(f"  device={args.device}, dtype={args.dtype}, attn={args.attn}")
    model = VGGT.from_pretrained(args.model, fused_attn=fused_attn).to(device=device, dtype=dtype).eval()

    if args.images:
        image_paths = args.images
        images = load_and_preprocess_images(image_paths, image_size=args.image_size)
        images = images.to(device=device, dtype=dtype)
        print(f"Loaded {len(image_paths)} images: {[os.path.basename(p) for p in image_paths]}")
    else:
        print(f"Using {args.num_images} dummy images (no --images provided).")
        images = torch.rand(args.num_images, 3, args.image_size, args.image_size, device=device, dtype=dtype)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=dtype).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=dtype).view(1, 3, 1, 1).to(device)
        images = (images - mean) / std

    print("\nRunning inference...")
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(images)
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - t0
    print(f"Inference done in {elapsed:.3f}s ({elapsed / len(images):.3f}s per image)")

    pred_cam = out["pose_enc"]
    pred_points = out["world_points"]
    pred_conf = out["world_points_conf"]

    print(f"\nOutput shapes:")
    print(f"  pred_cam (camera params):   {list(pred_cam.shape)}")
    print(f"  pred_points (3D point maps): {list(pred_points.shape)}")
    print(f"  pred_conf (confidence):      {list(pred_conf.shape)}")

    torch.save(pred_cam.cpu(), out_dir / "pred_cam.pt")
    torch.save(pred_points.cpu(), out_dir / "pred_points.pt")
    torch.save(pred_conf.cpu(), out_dir / "pred_conf.pt")

    summary = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "attn": args.attn,
        "num_images": len(images),
        "inference_time_s": round(elapsed, 3),
        "time_per_image_s": round(elapsed / len(images), 3),
        "output_dir": str(out_dir.resolve()),
        "outputs": {
            "pred_cam_shape": list(pred_cam.shape),
            "pred_points_shape": list(pred_points.shape),
            "pred_conf_shape": list(pred_conf.shape),
        }
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved outputs to {out_dir.resolve()}:")
    print(f"  pred_cam.pt        — camera intrinsics/extrinsics per view")
    print(f"  pred_points.pt     — 3D point maps (H x W x 3 per view)")
    print(f"  pred_conf.pt       — confidence maps")
    print(f"  summary.json       — inference metadata")

    if args.save_vis:
        try:
            from PIL import Image
            import numpy as np
            for i in range(len(images)):
                depth = 1.0 / (pred_points[i, :, :, 2].cpu().float().numpy() + 1e-6)
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
                depth_vis = (depth * 255).astype(np.uint8)
                Image.fromarray(depth_vis, mode="L").save(out_dir / f"depth_{i:04d}.png")
            print(f"  depth_*.png        — inverse-depth visualizations")
        except ImportError:
            print("  (PIL not available, skipping depth visualizations)")


if __name__ == "__main__":
    main()
