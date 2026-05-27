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

### Why Quantization is Challenging on AMD

Popular quantization libraries are designed for NVIDIA GPUs and lack proper AMD support:

| Library | Issue | Status |
|---------|-------|--------|
| **bitsandbytes** | Pre-compiled binaries only for ROCm 6.2–7.2; newer versions not provided | ❌ Requires source compilation |
| **torchao** | C++ extensions (Cutlass kernels) are CUDA-only; PyTorch fallback paths lack quantization implementations | ❌ Not compatible with ROCm |
| **quanto** | Early versions (0.2.x) had API bugs (weights reset to None during quantization) | ⚠️ Possible in newer versions but untested |
