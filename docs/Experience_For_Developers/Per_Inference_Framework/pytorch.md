# PyTorch Inference on AMD Ryzen AI / APU

## Overview

PyTorch with native ROCm backend is the **recommended approach** for VLM inference on AMD APU systems. It provides:

- Stable, well-documented API with no package conflicts
- GPU acceleration for both vision encoder and decoder
- No complex version matching or custom compilation required

---

## Native PyTorch ROCm Setup

### Basic Installation

```bash
# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

For newer ROCm versions, check [PyTorch's official installation guide](https://pytorch.org/get-started/locally/) for the appropriate index URL.

### Verification

```python
import torch
print(torch.cuda.is_available())        # Should be True
print(torch.cuda.get_device_name(0))   # Should show your GPU
print(torch.cuda.get_device_capability(0))  # Should show compute capability
```

---

## Quantization on PyTorch + ROCm

### Working Approach: AMD Quark

[AMD Quark](https://quark.docs.amd.com/latest/) is AMD's official quantization toolkit and works on ROCm. It supports **Eager Mode PTQ** with INT8, INT4, FP8 data types.

**Key Findings (ROCm 7.13 + SmolVLM-Instruct):**

| Method | Quant Time | Memory | Status |
|--------|-----------|--------|--------|
| INT8 weight-only (per-tensor) | ~0.5s | 4.18 GB (on-the-fly) | ✅ Works |
| INT4 weight-only (group=32) | ~3.7s | 4.18 GB (on-the-fly) | ✅ Works |

> **Note:** Weight-only quantization with Quark stores weights in original dtype + scale/zero-points, quantizing/dequantizing on-the-fly during computation. For actual memory reduction, use `quark.torch.export_safetensors()` to export compressed format.

**Usage:**

```python
from quark.torch.quantization.config.config import QConfig, QLayerConfig
from quark.torch.quantization import Int4PerGroupSpec
from quark.torch import ModelQuantizer

w4_spec = Int4PerGroupSpec(
    ch_axis=0, group_size=32,
    symmetric=True, scale_type="float",
    round_method="half_even", is_dynamic=False,
).to_quantization_spec()

quant_config = QConfig(
    global_quant_config=QLayerConfig(weight=w4_spec),
    exclude=["lm_head", "model.vision_model.*", "model.connector.*"],
)

quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model)
```

**Important:** The vision encoder (`model.vision_model.*`) and connector (`model.connector.*`) must be excluded from quantization to avoid Conv2D compatibility issues.

See `~/capstone/scripts/quantize_smolvlm.py` for a complete working example.

### Other Libraries

| Library | Issue | Status |
|---------|-------|--------|
| **bitsandbytes** | Pre-compiled binaries only for ROCm 6.2–7.2; on 7.13 requires source compilation or symlink workaround | ⚠️ Symlink to `rocm72` loads but kernels fail |
| **torchao** | C++ extensions (Cutlass kernels) are CUDA-only; PyTorch fallback paths lack quantization implementations | ❌ Not compatible with ROCm |
| **quanto** | v0.2.0 is latest; may have API bugs | ⚠️ Untested on this platform |
