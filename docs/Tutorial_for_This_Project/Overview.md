# Overview

This project benchmarks **SmolVLM** (vision-language) and **VGGT** (3D vision) on the AMD Ryzen 7 8845HS APU (Radeon 780M, RDNA 3, gfx1103). The goal: identify practical deployment frameworks for on-device AI perception.

---

## Hardware & Environment

| Component | Detail |
|---|---|
| **Device** | Beelink SER Mini PC |
| **APU** | AMD Ryzen 7 8845HS (8C/16T, Zen 4) |
| **iGPU** | AMD Radeon 780M (RDNA 3, gfx1103, 12 CU) |
| **OS** | Ubuntu 24.04 LTS (headless, no GUI) |
| **Kernel** | 6.17.0 |
| **ROCm** | 7.13.0a (TheRock community build) |

!!! warning "ROCm Support Status"
    The Ryzen 7 8845HS is not officially supported by AMD ROCm for native Linux. GPU driver uses the open-source `amdgpu` kernel module via TheRock.

    **However**, since [librocdxg v1.2.0](https://github.com/ROCm/librocdxg/releases/tag/v1.2.0) (May 2026), the WSL2 GPU passthrough path is **officially supported** — `gfx1103` is in librocdxg's supported device list.

    **Important for model selection**: On native Linux, TheRock's MIOpen (convolution library) has incomplete kernel databases for `gfx1103`. In contrast, WSL2 uses [librocdxg](https://github.com/ROCm/librocdxg) to bridge GPU compute to the production-grade Windows AMD driver, which ships with a complete MIOpen kernel database. This causes a dramatic performance gap for Conv-heavy models (see VGGT results below), while Transformer-heavy models (SmolVLM) see no difference.

---

## Testing Logic

All benchmarks follow the methodology documented in [Profiling_Methodology.md](Profiling_Methodology.md). Key parameters:

- **Input size**: 2 images × 384px (SmolVLM) / 518px (VGGT)
- **Warmup**: 10 iterations
- **Test iterations**: 20 latency runs, 10 power runs
- **Power measurement**: RAPL on Linux (CPU+iGPU package), subtracted idle baseline

Three environments were benchmarked: **Windows** (native ROCm), **WSL 2** (GPU passthrough), **Linux** (native ROCm via TheRock).

---

## Model Comparison & Recommendations

| Model | Size | Best Config | TTFT / Latency | Peak Memory | Framework |
|---|---|---|---|---|---|
| **SmolVLM2-2.2B-Instruct** | 2.2B | llama.cpp Vulkan f16 | ~116ms TTFT, ~47ms TPOT | ~0.2 GB | llama.cpp (recommended) / PyTorch (full precision only) |
| **SmolVLM-Instruct** | 2.2B | llama.cpp Vulkan f16 | ~116ms TTFT, ~47ms TPOT | ~0.2 GB | llama.cpp (patched) / PyTorch BF16 SDPA |
| **SmolVLM-500M-Instruct** | 500M | llama.cpp Vulkan f16 | ~42ms TTFT, ~11ms TPOT | ~0.14 GB | llama.cpp / PyTorch BF16 |
| **SmolVLM-256M-Instruct** | 256M | llama.cpp Vulkan f16 | ~21ms TTFT, ~5ms TPOT | ~0.13 GB | llama.cpp / PyTorch BF16 |
| **VGGT-1B** | 1B | WSL PyTorch iGPU BF16-SDPA | ~1.56s/image (WSL) / ~30.3s/image (Linux) | ~2.84 GB (WSL) / ~2.71 GB (Linux) | PyTorch only (WSL: fast; Linux: slow CPU-bound fallback) |

### Key Decisions

1. **SmolVLM — use llama.cpp on Vulkan**: f16 quantization achieves real-time inference (~116ms TTFT) with minimal memory footprint. Q8_0 is nearly as fast with slightly lower memory. PyTorch BF16 is only viable when full precision is required.
2. **SmolVLM-Instruct — prefer SmolVLM2-2.2B-Instruct**: the original model requires a tokenizer patch in llama.cpp; SmolVLM2 has proper support upstream.
3. **VGGT — PyTorch on WSL**: VGGT is fundamentally CPU-bound. WSL (sharing Windows GPU driver) delivers ~20× better throughput than native Linux (~1.56s vs ~30.3s per image with BF16-SDPA). Linux native PyTorch with BF16 is the only self-hosted option but is slow (~30s/image).
4. **Avoid ROCm backend in llama.cpp**: ROCm performs on par with CPU but draws significantly more power (~105W vs ~48W) with no speed advantage.
