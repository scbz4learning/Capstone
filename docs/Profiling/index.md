# Profiling & Benchmarking Results

## Overview

This section documents performance profiling for **SmolVLM** (vision-language) and **VGGT** (3D vision) models on the **AMD Ryzen 7 8845HS APU (Radeon 780M)** across three environments: Windows, WSL, and Linux.

!!! info "Navigation"
    - [Methodology](methodology.md) — Hardware, software, metrics, measurement methods
    - [Full Results](result.md) — All data tables and charts

---

## Key Findings

### SmolVLM-Instruct (2.2B) — PyTorch

**iGPU with SDPA attention is the optimal PyTorch configuration**, delivering consistent ~7s TTFT across all platforms while using 35% less VRAM than Eager attention.

| Metric | CPU-BF16 | iGPU-BF16-Eager | iGPU-BF16-SDPA |
|---|---|---|---|
| TTFT (ms) | ~108K–153K | ~12.1K–12.4K | **~6.9K–7.3K** |
| TPOT (ms/tok) | ~147–288 | ~111–122 | **~99–107** |
| Peak Memory (GB) | 5.1–6.3 | 7.85 | **5.09** |
| Tokens per Joule | 0.006–0.019 | 0.088–0.119 | **0.117–0.165** |

!!! tip "iGPU over CPU"
    iGPU acceleration provides **20–22× faster TTFT** and **1.5–2.7× faster TPOT** compared to CPU-only inference, while consuming comparable or lower power.

### llamesa.cpp — Projected Impact (Pending Retest)

!!! abstract "llama.cpp Will Be a Game Changer"
    While PyTorch iGPU delivers ~7s TTFT, llama.cpp on the **same hardware** with quantized models is projected to achieve **real-time inference** (< 100ms TTFT, ~15ms TPOT), based on results from smaller SmolVLM models:

    | Model | llama.cpp Vulkan TTFT | PyTorch iGPU TTFT | Speedup |
    |---|---|---|---|
    | SmolVLM2-2.2B | **95 ms** (Q4_K_M) | 6619 ms | **~70×** |
    | SmolVLM-500M | **40 ms** (Q8_0) | 2422 ms | **~60×** |
    | SmolVLM-256M | **20 ms** (f16) | 2133 ms | **~107×** |

    These speedups come with quantization-induced accuracy loss that needs evaluation (see [Planned Accuracy Benchmarks](methodology.md#accuracy-benchmarks-planned)).

!!! warning "llama.cpp SmolVLM-Instruct — Requires Custom Build"
    The **full SmolVLM-Instruct (2.2B) model** cannot run with the **official llama.cpp** release due to model architecture compatibility issues. A custom patch is being developed.

    **Recommended alternatives while patch is in progress:**
    - **SmolVLM2-2.2B-Instruct** (fully compatible, similar quality, ~95ms TTFT via Vulkan Q4_K_M)
    - **Smaller models** (256M / 500M) via llama.cpp for real-time use cases
    - **PyTorch with ROCm** for full-precision inference

### VGGT

!!! warning "VGGT is CPU-Bound"
    VGGT shows **marginal GPU benefit** due to its model architecture. The fastest configuration is iGPU-BF16 on Linux at **30.3s per image** (0.066 img/s).

| Config | Latency | Throughput | Note |
|---|---|---|---|
| CPU (any platform) | 32s–327s | 0.006–0.061 img/s | Linux fastest, WSL slowest |
| iGPU BF16 (Linux) | **30.3s** | **0.066 img/s** | Best result |
| iGPU F32 (Linux) | 43.3s–45.1s | 0.044–0.046 img/s | Negligible SDPA benefit |

    iGPU results on Windows/WSL are incomplete — see [VGGT section](result.md#3-vggt) for details.

### Platform Comparison

| Workload | Windows | WSL | Linux |
|---|---|---|---|
| CPU inference | Slowest | Slow | **Fastest (15–30% better)** |
| iGPU inference | ✅ Good | ✅ Good | ✅ Good (±5% across platforms) |
| Power measurement | Direct RAPL | Host-side RAPL | Direct RAPL |

!!! note "WSL Power Caveat"
    WSL GPU power is measured indirectly from the Windows host. Efficiency metrics (Tokens/Joule, img/W) may not be directly comparable to native Windows/Linux measurements.

### Model Size vs Speed (llama.cpp, Linux)

![SmolVLM llama.cpp TTFT](../assets/profiling/smolvlm_llamacpp_ttft_ms.png)

The chart above illustrates the dramatic speed difference across model sizes and backends. SmolVLM-256M via Vulkan achieves **real-time inference** at only **20ms TTFT**, making it suitable for interactive robotics applications where latency matters more than peak accuracy.

---

## Quick Navigation

| Section | Content |
|---|---|
| [Methodology](methodology.md) | Hardware specs, software versions, metric definitions |
| [SmolVLM-Instruct (PyTorch)](result.md#1-smolvlm-instruct-pytorch) | Full precision results across 3 environments |
| [SmolVLM-Instruct (llama.cpp, Pending)](result.md#2-smolvlm-instruct-llamacpp--pending) | Placeholder for upcoming llama.cpp retest |
| [VGGT](result.md#3-vggt) | All configs across platforms |
| [Other SmolVLM Models](result.md#4-other-smolvlm-family-models) | Smaller models, PyTorch + llama.cpp |

## Next Steps

!!! example "Planned Work"
    1. **Retest llama.cpp SmolVLM-Instruct** on Linux (CPU + Vulkan + ROCm)
    2. **Test llama.cpp on Windows** (CPU + Vulkan) and **WSL** (CPU + ROCm)
    3. **Evaluate quantization accuracy** on TextVQA and AI2D benchmarks
    4. **Complete VGGT iGPU testing** on Windows/WSL