# IREE GPU Deployment Guide

IREE (Intermediate Representation Execution Environment) is an MLIR-based compiler and runtime that supports multiple hardware backends. This guide documents common issues and solutions when deploying VLMs and 3D reconstruction models on AMD GPUs.

## 1. IREE GPU Backend Support Matrix

### Feature Availability by Backend

| Capability | ROCm (HIP) | Vulkan SPIR-V |
|------------|:----------:|:----------:|
| **Driver Detection** | ✅ | ✅ |
| **Device Recognition** | ✅ | ✅ |
| **Basic Ops Compilation** | ✅ | ✅ |
| **Simple Model Inference** | ✅ | ✅ |
| **VLM Text Decoder** | ✅ | ❌ |
| **`online_attention` Op** | ✅ | ❌ |
| **Vision Encoder Export** | ❌ | ❌ |
| **Full Model Export** | ❌ | ❌ |
| **Autoregressive Generation** | ❌ | ❌ |
| **Multimodal Models** | ❌ | ❌ |
| **3D Models (VGGT)** | ❌ | ❌ |

### Backend Capabilities Detail

**ROCm (HIP) Backend**:
- ✅ Driver and device discovery works correctly
- ✅ Can compile and run simple operators and small models
- ✅ Successfully compiles text decoder modules to GPU code
- ✅ `online_attention` MLIR ops are supported in AMDGPU code generation
- ❌ Cannot export full VLM or multimodal models due to PyTorch export limitations
- ❌ Autoregressive generation requires external implementation

**Vulkan SPIR-V Backend**:
- ✅ Driver and device discovery works
- ✅ Can compile basic MLIR operations
- ❌ `online_attention` ops cannot be lowered to SPIR-V (workgroup memory type mismatch)
- ❌ Cannot handle text decoder modules requiring modern attention operations
- ❌ Cannot export complex models

### Conclusion

**IREE on AMD GPU is currently not production-ready** for VLM and multimodal model deployment. While the runtime and basic compilation infrastructure work correctly, the following blockers prevent practical use:

1. **PyTorch Export Incompatibilities**: Modern transformers use control structures and higher-order ops that IREE cannot handle, affecting all backends
2. **Vulkan Backend Limitations**: Cannot handle `online_attention` ops needed for attention mechanisms
3. **Architectural Mismatch**: IREE is designed for static graphs, but VLMs require dynamic control flow, KV cache management, and flexible generation loops

**Current Status**: Do not use IREE for VLM inference. Use native PyTorch ROCm instead.


---

## 2. Known Deployment Issues

### PyTorch Export Limitations

When exporting VLM models to IREE, the following issues occur across all backends:

- **HuggingFace forward incompatibility**: transformers' forward wraps code in `torch.no_grad()`, producing `wrap_with_set_grad_enabled` ops that IREE FxImporter cannot handle
- **Dynamic shape guards**: `torch.nn.functional.scaled_dot_product_attention` contains data-dependent checks that `torch.export()` cannot handle
- **Vision encoder loops**: Vision transformers process patches via Python loops, which `torch.export` cannot trace statically
- **Autoregressive generation**: `model.generate()` requires external KV cache and sampling logic; IREE only accelerates individual forward passes

### Vulkan Backend Specific

- `online_attention` ops produce workgroup memory types that cannot be lowered to SPIR-V
- ROCm HIP backend does not have this limitation

---

## 3. Model Test Results

### SmolVLM
- ✅ Native PyTorch ROCm inference works
- ✅ Text decoder can be compiled to ROCm code (as isolated module)
- ❌ Full model export fails due to PyTorch bridge issues
- ❌ Generation loop requires external implementation

### VGGT (3D Reconstruction)
- ✅ Native PyTorch ROCm inference works
- ❌ Cannot export due to RoPE layer using data-dependent shape guards
- Same root cause as SmolVLM export failures

---

## 4. Deployment Options

### Option 1: Native PyTorch ROCm (Recommended)
```python
model = TransformerModel.from_pretrained("...").to("cuda")
output = model.generate(**inputs, max_new_tokens=20)
```
- ✅ Works for all models and use cases
- ✅ Best debugging and flexibility
- ❌ No IREE compilation benefits
- ❌ ROCm only

### Option 2: IREE with Custom Loop (Experimental)
For models that can be decomposed into simple submodules:
```bash
# Export only text decoder (no complex control flow)
iree-compile decoder.mlir --iree-hal-target-backends=rocm --iree-rocm-target=gfxXXXX -o decoder.vmfb

# Implement generation loop manually in application
```
- ✅ Single-step inference on GPU
- ❌ Requires manual generation loop implementation
- ❌ Cannot handle multimodal or complex architectures

### Option 3: amdsharktank Paged LLM
https://github.com/nod-ai/amd-shark-ai provides specialized LLaMA export:
- ✅ Works for pure text models (LLaMA, Qwen)
- ❌ Multimodal models not supported

---

## 5. Environment Setup

### Configure IREE for ROCm HIP

```bash
#!/bin/bash
# fix_iree_rocm_env.sh - Configure IREE ROCm environment

VENV="${VIRTUAL_ENV:-.venv}"
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES="$VENV/lib/python$PYVER/site-packages"

# Locate all ROCm SDK library directories
ALL_LIB_DIRS=()
for d in "$SITE_PACKAGES"/_rocm_sdk_*/lib; do
    [ -d "$d" ] && ALL_LIB_DIRS+=("$d")
done

# Fix missing .so symlinks
for d in "${ALL_LIB_DIRS[@]}"; do
    for f in "$d"/lib*.so.*; do
        [ -f "$f" ] || continue
        base="${f%%.so*}.so"
        [ -e "$base" ] || [ -L "$base" ] || ln -sf "$(basename "$f")" "$base"
    done
done

# Set environment variables
export LD_LIBRARY_PATH="${ALL_LIB_DIRS[0]}:$LD_LIBRARY_PATH"
export IREE_HIP_DYLIB_PATH="${ALL_LIB_DIRS[0]}"

echo "IREE_HIP_DYLIB_PATH=${ALL_LIB_DIRS[0]}"
python3 -c "
import iree.runtime as rt
driver = rt.get_driver('hip')
print('Available HIP devices:', driver.query_available_devices())
"
```

**Usage**: `source fix_iree_rocm_env.sh`

---

## References

- [IREE ROCm Backend Documentation](https://iree.dev/guides/deploying-on-amd/)
- [IREE Turbine AOT Export](https://github.com/iree-org/iree/tree/main/compiler/plugins/input/Torch)
- [amdsharktank Paged LLM](https://github.com/nod-ai/amd-shark-ai)
- [PyTorch torch.export Limitations](https://pytorch.org/docs/stable/generated/torch.export.export.html)
