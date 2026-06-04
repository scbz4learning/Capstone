# VGGT: All Outputs Explained

VGGT jointly predicts **6 types of 3D attributes** from a set of images in a single feed-forward pass.

---

## 1. Camera Parameters

| Output | Description |
|--------|-------------|
| **Intrinsics** (`fx`, `fy`, `cx`, `cy`) | Focal length & principal point in pixels, recovered per frame |
| **Extrinsics** (`R|t`, 3×4 matrix) | Camera pose in world coordinates (OpenCV convention: camera-from-world) |

## 2. Depth Map

Per-pixel metric depth prediction with confidence scores.

| Output | Shape | Meaning |
|--------|-------|---------|
| `depth_map` | `[S, H, W, 1]` | Depth from camera to each pixel (meters) |
| `depth_conf` | `[S, H, W]` | Confidence per pixel; higher = more reliable |

## 3. Point Map (Point Head)

Direct per-pixel 3D coordinate regression from learned features.

| Output | Shape | Meaning |
|--------|-------|---------|
| `point_map` | `[S, H, W, 3]` | (X, Y, Z) world coordinates for each pixel |
| `point_conf` | `[S, H, W]` | Confidence per 3D point |

## 4. Unprojected Point Map (Depth + Camera)

Combines depth map with camera extrinsics/intrinsics via geometric unprojection.  
**Typically more accurate** than the Point Head branch since it enforces multi-view geometry.

## 5. Point Tracks (Track Head)

Given user-specified query pixels `(x, y)`, the model tracks them across all frames.

| Output | Shape | Meaning |
|--------|-------|---------|
| `track_list` | `[S, N, 2]` | Tracked 2D positions for each query point in each frame |
| `vis_score` | `[S, N]` | Visibility per point per frame (0 = occluded / out of frame) |
| `conf_score` | `[N]` | Overall tracking confidence per query point |

---

## Inference Pipeline

```
images [S, 3, H, W]
        │
        ▼
   model.aggregator()
        │
        ├──► camera_head  ──► pose_enc ──► extrinsic + intrinsic
        ├──► depth_head   ──► depth_map, depth_conf
        ├──► point_head   ──► point_map, point_conf
        ├──► unprojection ──► more accurate 3D point cloud
        └──► track_head   ──► track_list, vis_score, conf_score
```

## Key Insight

All 5 heads share the same aggregated tokens (no re-encoding), enabling **1 GPU-second** reconstruction from any number of views.
