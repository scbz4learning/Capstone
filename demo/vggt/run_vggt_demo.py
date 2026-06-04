import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third-party" / "vggt"))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def extract_keyframes(video_path, out_dir, num_frames=4):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    indices = [int(i * total / (num_frames + 1)) for i in range(1, num_frames + 1)]
    frames_dir = out_dir / "input_frames"
    # Clean old frames before extracting new ones
    if frames_dir.exists():
        for old_file in frames_dir.iterdir():
            old_file.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            ts = idx / fps
            p = frames_dir / f"frame_{idx:04d}_t{ts:.1f}s.png"
            cv2.imwrite(str(p), frame)
            paths.append(p)
    cap.release()
    print(f"Extracted {len(paths)} frames from {total} frames ({total/fps:.1f}s)")
    return sorted(paths)


@torch.no_grad()
def run_vggt(frame_paths, device, dtype, out_dir):
    model = VGGT.from_pretrained("/home/bokai/capstone/models/VGGT-1B", fused_attn=True)
    model = model.to(device=device, dtype=dtype).eval()
    print(f"VGGT loaded on {device}, dtype={dtype}")

    images = load_and_preprocess_images([str(p) for p in frame_paths])
    images = images.to(device=device, dtype=dtype)
    H, W = images.shape[-2:]
    S = images.shape[0]
    print(f"Input tensor: {images.shape}  ({S} frames, {H}x{W})")

    images_batch = images[None]  # [1, S, 3, H, W]

    t0 = time.perf_counter()

    with torch.cuda.amp.autocast(dtype=dtype):
        aggregated_tokens_list, ps_idx = model.aggregator(images_batch)

        # ==================== 1. Camera Prediction ====================
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]  # [1, S, 9]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (H, W))
        # extrinsic: [1, S, 3, 4]  (OpenCV: camera from world)
        # intrinsic: [1, S, 3, 3]

        # ==================== 2. Depth Prediction ====================
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
        # depth_map:  [1, S, H, W, 1]
        # depth_conf: [1, S, H, W]

        # ==================== 3. Point Map Prediction ====================
        point_map, point_conf = model.point_head(aggregated_tokens_list, images_batch, ps_idx)
        # point_map:  [1, S, H, W, 3]
        # point_conf: [1, S, H, W]

        # ==================== 4. Unprojected Point Map (more accurate) ====================
        # Convert to float32 for numpy compatibility (bfloat16 is not numpy-serializable)
        point_map_by_unprojection = unproject_depth_map_to_point_map(
            depth_map.squeeze(0).float(), extrinsic.squeeze(0).float(), intrinsic.squeeze(0).float()
        )
        # unproject returns numpy; convert back to torch for consistency
        if isinstance(point_map_by_unprojection, np.ndarray):
            point_map_by_unprojection = torch.from_numpy(point_map_by_unprojection.copy()).to(device)
        # [S, H, W, 3]

        # ==================== 5. Track Prediction ====================
        grid_x = torch.linspace(W * 0.15, W * 0.85, 4)
        grid_y = torch.linspace(H * 0.15, H * 0.85, 4)
        query_points = torch.stack(torch.meshgrid(grid_x, grid_y, indexing="xy"), dim=-1).reshape(-1, 2).to(device)
        # query_points: [16, 2]

        track_list, vis_score, conf_score = model.track_head(
            aggregated_tokens_list, images_batch, ps_idx, query_points=query_points[None]
        )
        # track_head returns coord_preds as a list over iterations; take the final refinement
        if isinstance(track_list, list):
            track_list = track_list[-1]
        # track_list: [1, S, 16, 2] (2D tracked positions)
        # vis_score:  [1, S, 16]   (visibility per point per frame)
        # conf_score: [1, 16]       (overall confidence per point)

    elapsed = time.perf_counter() - t0
    print(f"Inference: {elapsed:.2f}s total, {elapsed/S:.2f}s per frame")

    # Save all tensors
    torch.save(pose_enc.cpu(), out_dir / "pose_enc.pt")
    torch.save(depth_map.cpu(), out_dir / "depth.pt")
    torch.save(depth_conf.cpu(), out_dir / "depth_conf.pt")
    torch.save(point_map.cpu(), out_dir / "pred_points.pt")
    torch.save(point_conf.cpu(), out_dir / "pred_conf.pt")
    torch.save(extrinsic.cpu(), out_dir / "pred_cam.pt")  # [1, S, 3, 4]
    torch.save(intrinsic.cpu(), out_dir / "pred_intri.pt")  # [1, S, 3, 3]
    torch.save(point_map_by_unprojection.cpu(), out_dir / "unproject_points.pt")
    torch.save(track_list.cpu(), out_dir / "track_list.pt")
    torch.save(vis_score.cpu(), out_dir / "track_vis.pt")
    torch.save(conf_score.cpu(), out_dir / "track_conf.pt")
    torch.save(query_points.cpu(), out_dir / "query_points.pt")

    preproc_H, preproc_W = images.shape[-2:]

    return {
        "pose_enc": pose_enc,
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "depth_map": depth_map,
        "depth_conf": depth_conf,
        "point_map": point_map,
        "point_conf": point_conf,
        "point_map_by_unprojection": point_map_by_unprojection,
        "track_list": track_list,
        "vis_score": vis_score,
        "conf_score": conf_score,
        "query_points": query_points,
        "elapsed": elapsed,
        "preproc_H": preproc_H,
        "preproc_W": preproc_W,
    }


def save_camera_visualizations(frame_paths, extrinsic, intrinsic, out_dir):
    cam_dir = out_dir / "camera_views"
    cam_dir.mkdir(exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — register 3d
    except ImportError:
        print("matplotlib not available, skipping camera visualizations")
        return

    ext = extrinsic[0].cpu().float().numpy()  # [S, 3, 4]
    intr = intrinsic[0].cpu().float().numpy()  # [S, 3, 3]
    S = ext.shape[0]

    # Compute camera centers: center = -R^T @ t
    R = ext[:, :, :3]  # [S, 3, 3]
    t_vec = ext[:, :, 3]  # [S, 3]
    centers = -np.einsum('sij,sj->si', R.transpose(0, 2, 1), t_vec)  # [S, 3]

    # Forward directions (Z-axis of camera frame in world coords)
    forwards = R.transpose(0, 2, 1)[:, :, 2]  # [S, 3]

    # --- Top-down view (XZ plane) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(S):
        ax.scatter(centers[i, 0], centers[i, 2], s=120, label=f"Frame {i+1}")
        ax.annotate(f"F{i+1}", (centers[i, 0], centers[i, 2]),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)
        ax.arrow(centers[i, 0], centers[i, 2],
                 forwards[i, 0] * 0.5, forwards[i, 2] * 0.5,
                 head_width=0.15, head_length=0.15, fc="red", ec="red", alpha=0.7)
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Z (world)")
    ax.set_title("Camera Positions — Top-down View\n(dots: position, arrows: viewing direction)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(cam_dir / "camera_topdown.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- 3D view ---
    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(111, projection="3d")
    for i in range(S):
        ax3d.scatter(centers[i, 0], centers[i, 1], centers[i, 2], s=80, label=f"Frame {i+1}")
        ax3d.text(centers[i, 0], centers[i, 1], centers[i, 2], f"F{i+1}", fontsize=9)
        ax3d.quiver(centers[i, 0], centers[i, 1], centers[i, 2],
                    forwards[i, 0] * 0.5, forwards[i, 1] * 0.5, forwards[i, 2] * 0.5,
                    color="red", arrow_length_ratio=0.2)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("Camera Poses — 3D View")
    fig.tight_layout()
    fig.savefig(str(cam_dir / "camera_3d.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Intrinsics table as JSON for web ---
    cameras_data = []
    for i in range(S):
        K = intr[i]
        # Compute camera center from extrinsics
        cx_w, cy_w, cz_w = float(centers[i, 0]), float(centers[i, 1]), float(centers[i, 2])
        cameras_data.append({
            "frame": i,
            "fx": round(float(K[0, 0]), 2),
            "fy": round(float(K[1, 1]), 2),
            "cx": round(float(K[0, 2]), 2),
            "cy": round(float(K[1, 2]), 2),
            "pos_x": round(cx_w, 3),
            "pos_y": round(cy_w, 3),
            "pos_z": round(cz_w, 3),
        })
    with open(out_dir / "cameras.json", "w") as f:
        json.dump(cameras_data, f, indent=2)
    print(f"Camera data saved to {cam_dir}")


def save_depth_visualizations(frame_paths, depth, out_dir):
    depth_dir = out_dir / "depth_maps"
    depth_dir.mkdir(exist_ok=True)
    for i, fp in enumerate(frame_paths):
        d = depth[0, i, :, :, 0].cpu().float().numpy()
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-6)
        d_vis = (d_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(depth_dir / f"depth_{i:04d}_color.png"), depth_color)
        Image.fromarray(d_vis, mode="L").save(str(depth_dir / f"depth_{i:04d}_gray.png"))
    print(f"Depth maps saved to {depth_dir}")


def save_confidence_visualizations(depth_conf, out_dir):
    conf_dir = out_dir / "confidence_maps"
    conf_dir.mkdir(exist_ok=True)
    for i in range(depth_conf.shape[1]):
        c = depth_conf[0, i].cpu().float().numpy()
        c_norm = (c - c.min()) / (c.max() - c.min() + 1e-6)
        c_vis = (c_norm * 255).astype(np.uint8)
        conf_color = cv2.applyColorMap(c_vis, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(str(conf_dir / f"conf_{i:04d}.png"), conf_color)
    print(f"Confidence maps saved to {conf_dir}")


def save_point_cloud_images(frame_paths, world_points, out_dir, label="PointHead"):
    pc_dir = out_dir / "pointcloud_views"
    pc_dir.mkdir(exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for i, fp in enumerate(frame_paths):
            pts = world_points[0, i].cpu().float().numpy()
            h, w, _ = pts.shape
            mask = (pts[:, :, 2] > 0) & (pts[:, :, 2] < 100)
            x, y, z = pts[mask, 0], pts[mask, 1], pts[mask, 2]

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(x[::8], y[::8], z[::8], c=z[::8], cmap="plasma", s=1, alpha=0.6)
            ax.set_title(f"3D Point Cloud [{label}] — frame {i+1}")
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            fig.colorbar(scatter, ax=ax, label="Z (depth)")
            fig.savefig(str(pc_dir / f"pc_{i:04d}.png"), dpi=120, bbox_inches="tight")
            plt.close(fig)
        print(f"Point cloud views saved to {pc_dir}")
    except ImportError:
        print("matplotlib not available, skipping 3D point cloud views")


def save_unproject_visualizations(frame_paths, unproject_points, out_dir):
    """3D visualizations from depth+camera unprojected point map (more accurate)."""
    pc_dir = out_dir / "unproject_views"
    pc_dir.mkdir(exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pts = unproject_points.cpu().float().numpy()  # [S, H, W, 3]
        for i, fp in enumerate(frame_paths):
            frame_pts = pts[i]
            mask = (frame_pts[:, :, 2] > 0) & (frame_pts[:, :, 2] < 100)
            x, y, z = frame_pts[mask, 0], frame_pts[mask, 1], frame_pts[mask, 2]

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(x[::8], y[::8], z[::8], c=z[::8], cmap="viridis", s=1, alpha=0.6)
            ax.set_title(f"Unprojected 3D Points (Depth+Cam) — frame {i+1}")
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            fig.colorbar(scatter, ax=ax, label="Z (depth)")
            fig.savefig(str(pc_dir / f"unproj_{i:04d}.png"), dpi=120, bbox_inches="tight")
            plt.close(fig)
        print(f"Unprojected point cloud views saved to {pc_dir}")
    except ImportError:
        print("matplotlib not available, skipping unprojected 3D views")


def save_track_visualizations(frame_paths, track_list, vis_score, query_points, preproc_H, preproc_W, out_dir):
    track_dir = out_dir / "track_views"
    track_dir.mkdir(exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping track visualizations")
        return

    t_list = track_list[0].cpu().float().numpy()  # [S, N, 2]
    v_score = vis_score[0].cpu().float().numpy()  # [S, N]
    S, N, _ = t_list.shape

    # Compute per-frame scale factors from preprocessed (518-wide) to original frame size
    scale_x_list = []
    scale_y_list = []
    crop_start_y_list = []  # center-crop start offset (0 if no crop)
    for fp in frame_paths:
        img_cv = cv2.imread(str(fp))
        H_orig, W_orig = img_cv.shape[:2]
        scale_x_list.append(W_orig / preproc_W)
        # Reverse the crop-mode preprocessing:
        # Step 1: resize so width = 518, height = round(H_orig * 518 / W_orig / 14) * 14
        # Step 2: if new_height > 518, center-crop to 518 (start_y = (new_height - 518) // 2)
        new_height_before_crop = round(H_orig * (preproc_W / W_orig) / 14) * 14
        if new_height_before_crop > preproc_W:  # preproc_W == 518
            crop_start = (new_height_before_crop - preproc_W) // 2
            scale_y_list.append(H_orig / new_height_before_crop)
            crop_start_y_list.append(crop_start)
        else:
            scale_y_list.append(H_orig / preproc_H)
            crop_start_y_list.append(0)

    # Distinct colors for each query point
    cmap = plt.cm.get_cmap("tab20", N)
    colors = (cmap(range(N))[:, :3] * 255).astype(int)

    for i, fp in enumerate(frame_paths):
        img_cv = cv2.imread(str(fp))
        if img_cv is None:
            continue
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img_rgb)

        sx = scale_x_list[i]
        sy = scale_y_list[i]
        crop_start = crop_start_y_list[i]

        for j in range(N):
            x_pre = t_list[i, j, 0]
            y_pre = t_list[i, j, 1] + crop_start  # reverse center-crop: y before crop = y_pre + start
            vis = v_score[i, j]
            if vis > 0.3:
                x_orig = x_pre * sx
                y_orig = y_pre * sy
                color = colors[j] / 255.0
                alpha = min(1.0, vis + 0.3)
                ax.plot(x_orig, y_orig, "o", color=color, markersize=10, alpha=alpha, markeredgewidth=1,
                        markeredgecolor="white")
                ax.plot(x_orig, y_orig, "o", color=color, markersize=4, alpha=alpha)

        ax.set_title(f"Tracked Points — frame {i+1}")
        ax.axis("off")
        fig.savefig(str(track_dir / f"tracks_{i:04d}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"Track visualizations saved to {track_dir}")

    # Also save a combined track trail plot (one figure showing tracks across frames)
    fig, ax = plt.subplots(figsize=(10, 8))
    # Use first frame as background (with partial opacity to see trails)
    first_img = cv2.imread(str(frame_paths[0]))
    first_rgb = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
    ax.imshow(first_rgb, alpha=0.4)

    sx0 = scale_x_list[0]
    sy0 = scale_y_list[0]
    crop_start0 = crop_start_y_list[0]

    for j in range(N):
        xs_pre = t_list[:, j, 0]
        ys_pre = t_list[:, j, 1]
        vis = v_score[:, j]
        color = colors[j] / 255.0
        # Map all coordinates to original space of first frame
        xs_orig = xs_pre * sx0
        ys_orig = (ys_pre + crop_start0) * sy0  # reverse center-crop
        visible_mask = vis > 0.3
        for s in range(S - 1):
            if visible_mask[s] and visible_mask[s + 1]:
                ax.plot([xs_orig[s], xs_orig[s + 1]], [ys_orig[s], ys_orig[s + 1]],
                        "-", color=color, linewidth=1, alpha=0.6)
        for s in range(S):
            if visible_mask[s]:
                marker = "o" if s == 0 else ("s" if s == S - 1 else ".")
                size = 10 if s == 0 else (8 if s == S - 1 else 4)
                ax.plot(xs_orig[s], ys_orig[s], marker, color=color, markersize=size, alpha=0.8)
    ax.set_title(f"Track Trails ({N} points across {S} frames)")
    ax.axis("off")
    fig.savefig(str(track_dir / "tracks_trails.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Track trail overview saved to {track_dir}")


def create_static_html(frame_paths, out_dir):
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VGGT 3D Driving Demo</title>
<style>
body { font-family: system-ui; background: #111; color: #eee; max-width: 1400px; margin: auto; padding: 20px; }
h1 { text-align: center; color: #4fc3f7; font-size: 20px; }
h2 { color: #81d4fa; border-bottom: 1px solid #333; padding-bottom: 6px; margin-top: 30px; font-size: 15px; }
h3 { color: #aaa; font-size: 13px; margin: 8px 0; }
.grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin: 12px 0; }
.card { background: #1a1a2e; border-radius: 8px; padding: 8px; text-align: center; }
.card img { width: 100%; border-radius: 4px; max-height: 280px; object-fit: contain; }
.card .label { font-size: 11px; color: #888; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.3px; }
.full { width: 100%; margin: 8px 0; border-radius: 6px; max-height: 500px; object-fit: contain; }
table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 13px; }
td, th { padding: 6px 10px; border: 1px solid #333; text-align: center; }
th { background: #1a3a5c; }
.stats { background: #1a1a2e; padding: 16px; border-radius: 8px; line-height: 1.8; text-align: center; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 900px) {
  .grid { grid-template-columns: 1fr 1fr; }
  .two-col { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<h1>VGGT 3D Scene Reconstruction — All Outputs</h1>
<p style="text-align:center;color:#aaa;">Camera · Depth · Point Cloud · Unprojected · Tracks</p>
"""

    # Stats
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            s = json.load(f)
        html += f"""<div class="stats">
  <b>Device:</b> {s["device"]} · <b>Dtype:</b> {s["dtype"]}<br>
  <b>Total time:</b> {s["inference_time_s"]:.2f}s · <b>Per frame:</b> {s["time_per_image_s"]:.2f}s · <b>Frames:</b> {s["num_images"]}
</div>"""

    # Camera info
    html += '<h2> Camera Parameters</h2>'
    html += '<h3>Intrinsics</h3>'
    html += '<table><tr><th>Frame</th><th>fx</th><th>fy</th><th>cx</th><th>cy</th><th>Center X</th><th>Center Y</th><th>Center Z</th></tr>'
    cameras_path = out_dir / "cameras.json"
    if cameras_path.exists():
        with open(cameras_path) as f:
            cams = json.load(f)
        for c in cams:
            html += f'<tr><td>{c["frame"]+1}</td><td>{c["fx"]}</td><td>{c["fy"]}</td><td>{c["cx"]}</td><td>{c["cy"]}</td><td>{c["pos_x"]}</td><td>{c["pos_y"]}</td><td>{c["pos_z"]}</td></tr>'
    html += '</table>'

    html += '<h3>Camera Poses</h3>'
    html += '<div class="two-col">'
    html += '<div class="card"><img class="full" src="camera_views/camera_topdown.png"><div class="label">Top-down (XZ plane)</div></div>'
    html += '<div class="card"><img class="full" src="camera_views/camera_3d.png"><div class="label">3D View</div></div>'
    html += '</div>'

    # Per-frame sections
    for i, fp in enumerate(frame_paths):
        html += f'<h2>Frame {i+1}</h2>\n<div class="grid">\n'

        html += f'<div class="card"><img src="input_frames/{fp.name}"><div class="label">Original</div></div>\n'
        html += f'<div class="card"><img src="depth_maps/depth_{i:04d}_color.png"><div class="label">Depth (Inferno)</div></div>\n'
        html += f'<div class="card"><img src="confidence_maps/conf_{i:04d}.png"><div class="label">Confidence (Viridis)</div></div>\n'

        html += f'<div class="card"><img src="pointcloud_views/pc_{i:04d}.png"><div class="label">Point Cloud (Point Head)</div></div>\n'
        html += f'<div class="card"><img src="unproject_views/unproj_{i:04d}.png"><div class="label">Point Cloud (Depth+Cam)</div></div>\n'
        html += f'<div class="card"><img src="track_views/tracks_{i:04d}.png"><div class="label">Tracks Overlay</div></div>\n'

        html += "</div>\n"

    # Track trail overview
    html += '<h2> Track Trails (all points across frames)</h2>'
    html += '<div class="card"><img class="full" src="track_views/tracks_trails.png"><div class="label">Colored lines show point movement across frames</div></div>'

    html += "\n</body></html>"
    (out_dir / "index.html").write_text(html)
    print(f"HTML report: {out_dir / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description="VGGT Driving Sequence 3D Demo — All Outputs")
    parser.add_argument("--video", default="vggt-demo.mp4")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(args.video)
    if not video_path.exists():
        video_path = Path(__file__).parent / args.video
    if not video_path.exists():
        print(f"Video not found: {video_path}"); sys.exit(1)

    frame_paths = extract_keyframes(video_path, out_dir, args.frames)

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    results = run_vggt(frame_paths, device, dtype, out_dir)

    save_camera_visualizations(frame_paths, results["extrinsic"], results["intrinsic"], out_dir)
    save_depth_visualizations(frame_paths, results["depth_map"], out_dir)
    save_confidence_visualizations(results["depth_conf"], out_dir)
    save_point_cloud_images(frame_paths, results["point_map"], out_dir)
    save_unproject_visualizations(frame_paths, results["point_map_by_unprojection"], out_dir)
    save_track_visualizations(frame_paths, results["track_list"], results["vis_score"],
                              results["query_points"], results["preproc_H"], results["preproc_W"], out_dir)

    summary = {
        "video": str(video_path), "device": args.device, "dtype": args.dtype,
        "num_images": len(frame_paths), "inference_time_s": round(results["elapsed"], 2),
        "time_per_image_s": round(results["elapsed"] / len(frame_paths), 2),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    create_static_html(frame_paths, out_dir)
    print(f"\nDone! Open {out_dir / 'index.html'} in a browser.")


if __name__ == "__main__":
    main()
