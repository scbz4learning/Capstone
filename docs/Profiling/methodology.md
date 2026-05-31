# Testing Methodology

## Hardware Platform

| Component | Detail |
|---|---|
| **Device** | Beelink SER Mini PC |
| **APU** | AMD Ryzen 7 8845HS (8C/16T, Zen 4) |
| **iGPU** | AMD Radeon 780M (RDNA 3, gfx1103, 12 CU) |

## Software Environment

| Component | Version |
|---|---|
| **OS (Linux)** | Ubuntu 24.04.4 LTS, Kernel 6.17.0-29-generic |
| **OS (Windows)** | Windows 11 |
| **WSL** | WSL 2 (Ubuntu 24.04) |
| **Python** | 3.12.3 |
| **PyTorch** | 2.10.0+rocm7.13.0a20260424 (Linux); Windows: TBD; WSL: TBD |
| **ROCm** | 7.13.0a (installed via PyPI — `rocm-sdk-core`, `rocm-sdk-libraries-gfx110X-all`) |
| **Transformers** | 5.8.1 |
| **llama.cpp** | b9357 (patched for SmolVLM-Instruct) |

!!! warning "No Official ROCm Support"
    The Ryzen 7 8845HS / Radeon 760M is **not officially supported** by AMD's ROCm. Community support is provided via [TheRock](https://github.com/ROCm/TheRock) project. GPU driver on Linux uses the open-source `amdgpu` kernel driver.

## llama.cpp Build Status

!!! bug "SmolVLM-Instruct Full Model"
    The **full SmolVLM-Instruct (2.2B) model** cannot be correctly inferred with the **official llama.cpp** build due to model architecture compatibility issues. A custom patch is being developed to add proper support. See pending [GitHub issue](#) (TBD).

    **Recommended alternatives while the patch is in progress:**
    - Use **SmolVLM2-2.2B-Instruct** (fully compatible, similar quality)
    - Use **smaller models** (SmolVLM-256M / 500M) via llama.cpp for real-time use cases
    - Use **PyTorch with ROCm** for full-precision inference

## Test Environments

Three environments were benchmarked:

- **Windows** — Native Windows 11. GPU operations via ROCm HIP SDK (PyTorch with ROCm backend).
- **WSL** — WSL 2 (Ubuntu 24.04), shares the Windows GPU driver. GPU operations via ROCm passthrough.
- **Linux** — Native Ubuntu 24.04. GPU operations via ROCm community build (TheRock). Power measured directly via RAPL.

## Models Tested

### SmolVLM Family

| Model | Params | Type | llama.cpp Compat |
|---|---|---|---|
| SmolVLM-256M-Instruct | 256M | Vision-Language | ✅ Yes |
| SmolVLM-500M-Instruct | 500M | Vision-Language | ✅ Yes |
| SmolVLM-Instruct | 2.2B | Vision-Language | ❌ Fork/Patch needed |
| SmolVLM2-256M-Video-Instruct | 256M | Vision-Language (Video) | ✅ Yes |
| SmolVLM2-500M-Video-Instruct | 500M | Vision-Language (Video) | ✅ Yes |
| SmolVLM2-2.2B-Instruct | 2.2B | Vision-Language (Video) | ✅ Yes |

### VGGT

| Model | Params | Type |
|---|---|---|
| VGGT-1B | 1B | 3D Vision (Pointmap) |

## Configuration Naming

Results use the following convention:

```
OS-Framework-Device-Dtype-Accelerator
```

Examples:
- `Windows-PyTorch-iGPU-BF16-SDPA` — Windows, PyTorch, iGPU, BFloat16, SDPA attention
- `Linux-llama.cpp-CPU-Q4_K_M` — Linux, llama.cpp, CPU, 4-bit K-quant
- `WSL-llama.cpp-iGPU-f16-Vulkan` — WSL, llama.cpp, iGPU, float16, Vulkan

### PyTorch Configs

| Field | Values |
|---|---|
| Device | `CPU`, `iGPU` (Radeon 760M via ROCm) |
| Dtype | `BF16`, `F32` |
| Attention | `Eager`, `SDPA`, `none` (CPU only) |

### llama.cpp Configs

| Field | Values |
|---|---|
| Device | `CPU`, `iGPU` |
| Quantization | `f16`, `Q8_0`, `Q4_K_M` |
| Backend | `Vulkan`, `ROCm` |

## Metrics

| Metric | Description |
|---|---|
| **TTFT (ms)** | Time to First Token — prefill latency |
| **TPOT (ms)** | Time Per Output Token — decode latency |
| **Latency (ms)** | End-to-end single image inference (VGGT) |
| **Throughput (img/s)** | Images per second (VGGT) |
| **Avg Power (W)** | Average system power during inference (RAPL) |
| **Energy per Inference (J)** | Total energy per inference |
| **Tokens per Joule** | Energy efficiency (SmolVLM) |
| **Efficiency (img/W)** | Images per Watt (VGGT) |
| **Peak Memory (GB)** | Peak memory (RSS for CPU, VRAM for GPU) |

## Measurement Method

- **Power**: RAPL on CPU/iGPU package. WSL measured from Windows host.
- **Idle power**: Subtracted from total (measured before each run).
- **Warmup**: 10 iterations.
- **Test iterations**: 20 latency, 10 power.
- **SmolVLM input**: 2 images × 384px, prompt "Describe the images briefly.", 128 output tokens.
- **VGGT input**: 2 images × 518px.
- **Scripts**: `comprehensive_profile_smolvlm.py`, `comprehensive_profile_vggt.py` (see `/benchmarks/`).

## Accuracy Benchmarks (Planned)

| Benchmark | Description | Est. Time (llama.cpp Q4) |
|---|---|---|
| **TextVQA Val** | Text-based VQA (~5000 images) | ~20 min |
| **AI2D** | Diagram understanding (~4000 images) | ~15 min |
| **Perplexity** | Quick quality proxy | < 5 min |