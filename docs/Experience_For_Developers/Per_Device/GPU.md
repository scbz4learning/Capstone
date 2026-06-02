# GPU ROCm Driver Stack Analysis: WSL vs Native Linux

This document explains a critical finding when running ROCm on AMD GPUs using [TheRock](https://github.com/ROCm/TheRock) community builds compared to the WSL2 + [librocdxg](https://github.com/ROCm/librocdxg) stack.

!!! tip "Official Support Update (May 2026)"
    [librocdxg v1.2.0](https://github.com/ROCm/librocdxg/releases/tag/v1.2.0) added official support for `gfx1103` (Radeon 780M / Ryzen 8845HS iGPU) and `gfx1152`. The WSL2 + librocdxg path is now an **officially supported** AMD feature for these GPUs.

---

## The Technology Stack

### Linux: TheRock (Community ROCm)

TheRock is a lightweight, open-source build platform for HIP and ROCm, described by AMD as in **"early preview state"**. It provides nightly ROCm + PyTorch wheels for hardware not covered by official ROCm releases.

Key components in the GPU compute stack:

| Layer | Component | Status in TheRock |
|---|---|---|
| Runtime | HIP / ROCr | Functional |
| BLAS | rocBLAS / hipBLAS | Well-optimized |
| Convolution | **MIOpen** | **Early preview — kernel DB may be incomplete** |
| Attention | Flash Attention / SDPA | Well-optimized |
| Framework | PyTorch | Functional |

MIOpen relies on a database of pre-tuned convolution kernels (`.fdb.txt` files). On TheRock for `gfx1103`, these kernel databases are often **missing or incomplete**, as shown by this runtime warning:

```
MIOpen(HIP): Warning [ParseAndLoadDb] File is unreadable:
  ".../miopen/db/gfx1103_6.HIP.fdb.txt"
MIOpen(HIP): Warning [OpenRuntimeLibraryForDevice]
  CK grouped conv library not found for device gfx1103:
  libMIOpenCKGroupedConv_gfx1103.so: cannot open shared object file
```

When a Conv kernel is missing, MIOpen falls back to:
1. Auto-tuning (slow, first-run penalty)
2. A generic fallback kernel (much slower than optimized)
3. In the worst case, the operation may silently promote to float32

### WSL2: librocdxg + Windows AMD Driver

[librocdxg](https://github.com/ROCm/librocdxg) is a user-mode library that enables ROCm functionality on WSL2 by routing GPU compute calls through the **Windows AMD display driver** via DXCore/DXG:

```
PyTorch → ROCm/HIP → librocdxg → DXCore → Windows AMD Driver → GPU
```

The Windows AMD driver is a **production-grade** driver that ships with a complete MIOpen kernel database covering all supported GPU architectures. This means:
- Conv2d/ConvTranspose2d operations run at full hardware speed
- No auto-tuning or fallback needed
- All operators have optimized MIOpen kernels

---

## Impact on Model Performance

The gap between the two stacks depends entirely on **how heavily the model uses convolution operations**.

### Conv-Heavy Models (e.g., VGGT)

VGGT uses a DPT (Dense Prediction Transformer) head that extensively uses `nn.Conv2d` and `nn.ConvTranspose2d` layers. For this model:

| Environment | BF16 Latency | vs WSL |
|---|---|---|
| WSL (Windows driver) | **1.6s** | 1× |
| Windows native (TheRock) | 30.0s | ~19× slower |
| Linux native (TheRock) | 30.3s | ~19× slower |

Both Linux and native Windows use the TheRock ROCm build with the same incomplete MIOpen. WSL uniquely benefits from the production Windows driver via librocdxg.

### Transformer-Heavy Models (e.g., SmolVLM)

SmolVLM is dominated by `nn.Linear` (GEMM) and `scaled_dot_product_attention`, with no Conv layers. These ops go through rocBLAS/hipBLAS and Flash Attention, which are well-optimized even in TheRock:

| Environment | BF16 TTFT | vs WSL |
|---|---|---|
| WSL (Windows driver) | 12.4s | 1× |
| Windows native (TheRock) | 12.2s | ~1× |
| Linux native (TheRock) | 12.1s | ~1× |

No significant difference because convolution is not on the critical path.

---

## Recommendations by Scenario

### You have access to Windows + WSL2
- **Preferred path for Conv-heavy models** (VGGT, CNN-based detectors, segmentation models)
- Install WSL2 + librocdxg + ROCm PyTorch nightlies
- See [GPU Inference guide](../General_Advice/Advice_Per_Device/2_gpu-inference.md#wsl2) for installation steps

### You only have headless Linux
- TheRock is your only GPU option
- Expect slower Conv performance, especially on first run (MIOpen auto-tuning)
- Consider Vulkan as an alternative if the inference framework supports it
- Test with your specific model — Transformer-heavy models have no penalty

### You need both Conv speed + headless Linux
- Consider using a supported AMD GPU (official ROCm) instead
- Or accept the TheRock performance and optimize around it (e.g., fuse Conv layers, reduce resolution)

---

## Verification Checklist

To determine if TheRock's MIOpen is the bottleneck on your system:

```bash
# Check for missing MIOpen kernel databases
python -c "
import torch
torch.ones(1).cuda()  # trigger MIOpen init
# Look for warnings about missing .fdb.txt or CK libraries
"

# Compare Conv vs GEMM throughput
python -c "
import torch, time
device = 'cuda'
# GEMM (Linear) — should be fast
x = torch.randn(1024, 1024, device=device)
w = torch.randn(1024, 1024, device=device)
t0 = time.time()
for _ in range(100): torch.mm(x, w)
torch.cuda.synchronize()
print(f'GEMM: {(time.time()-t0)/100*1000:.2f}ms')

# Conv2d — may be slow on TheRock
x = torch.randn(8, 256, 32, 32, device=device)
c = torch.nn.Conv2d(256, 256, 3, padding=1).to(device)
t0 = time.time()
for _ in range(100): c(x)
torch.cuda.synchronize()
print(f'Conv2d: {(time.time()-t0)/100*1000:.2f}ms')
"
```

If the Conv2d throughput is disproportionately slow compared to GEMM on Linux, but fast on WSL (same GPU), TheRock's MIOpen is the bottleneck.

---

## Reference Links

- [librocdxg — ROCm DirectX GPU interop](https://github.com/ROCm/librocdxg)
- [TheRock — HIP Environment and ROCm Kit](https://github.com/ROCm/TheRock)
- [TheRock Supported GPUs](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
- [WSL ROCDXG CI documentation](https://github.com/ROCm/TheRock/blob/main/docs/development/wsl_rocdxg.md)
- [Issue #36 — Windows platform support bringup](https://github.com/ROCm/TheRock/issues/36)
