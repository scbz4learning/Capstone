# VGGT

VGGT (Visual Geometry Group Transformer) is a feed-forward model developed by Facebook Research for **3D visual perception** — estimating camera parameters, depth maps, and point maps from a set of input images. Unlike generative VLMs, VGGT is a **discriminative 3D reconstruction model**, making it a different class of perception model.

## Transformer Usage

VGGT uses Vision Transformer (ViT) architecture but in a way that differs from standard LLM/VLM inference:

- The model processes multiple images jointly in a single forward pass, rather than autoregressively generating tokens
- The transformer is used for spatial understanding across image pairs, not for text generation
- This non-standard usage means that many general-purpose inference frameworks (vLLM, ONNX Runtime, etc.) **do not support VGGT out of the box**

## Framework Recommendation

| Framework | Recommendation | Reason |
|-----------|---------------|--------|
| **PyTorch** | ✅ Recommended | Only fully supported option. Official code + BF16 adaptation available |
| **ONNX Runtime** | ❌ Not recommended | No automated export tool works. Community exports exist but have fixed dimensions and inefficiency issues |

## PyTorch Notes

### BF16 Support

The official repository only supports **FP32** inference. We adapted it to use **BF16**:

- Inference and results appear **normal** with BF16
- Performance is significantly better than FP32 on both CPU and iGPU
- **Caveat**: For production use, we recommend **further validation** — test on your own scenes with fixed image dimensions and input sizes over extended runs

### Image Constraints

VGGT imposes a strict constraint: the **short edge of input images must be a multiple of 14**. The `load_and_preprocess_images` utility handles this automatically, but custom data pipelines must respect this.

## ONNX Runtime

- No automated tooling exists to convert VGGT to a working ONNX format
- Community-provided exports (e.g. [akretz/vggt-onnx](https://github.com/akretz/vggt-onnx)) have **fixed dimensions** that are difficult to modify and suffer from low efficiency
- See [ONNX page](1.onnxruntime.md) for attempted approaches

## Citation

```bibtex
@inproceedings{wang2026vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and others},
  booktitle={CVPR},
  year={2026}
}
```