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

### llama.cpp — Actual Results

!!! abstract "llama.cpp Delivers Real-Time Inference"
    llama.cpp with quantized models on the **same hardware** achieves **real-time inference** (< 100ms TTFT, ~15ms TPOT) for SmolVLM-Instruct, matching the performance previously seen on smaller models:

    | Model | llama.cpp Vulkan TTFT | PyTorch iGPU TTFT | Speedup |
    |---|---|---|---|
    | SmolVLM-Instruct (2.2B) | **95 ms** (Q4_K_M) | 6934 ms | **~73×** |
    | SmolVLM2-2.2B-Instruct | **95 ms** (Q4_K_M) | 6619 ms | **~70×** |
    | SmolVLM-500M | **40 ms** (Q8_0) | 2422 ms | **~60×** |
    | SmolVLM-256M | **20 ms** (f16) | 2133 ms | **~107×** |

    These speedups come with quantization-induced accuracy loss that needs evaluation (see [Planned Accuracy Benchmarks](methodology.md#accuracy-benchmarks-planned)).

!!! success "llama.cpp SmolVLM-Instruct — Fully Tested"
    The **full SmolVLM-Instruct (2.2B) model** has been successfully profiled across CPU, Vulkan, and ROCm backends and all three quantizations (f16, Q8_0, Q4_K_M). See the [Full Results](result.md#1-smolvlm-instruct) for detailed metrics.

!!! success "Key Findings"
    - **Vulkan Q4_K_M delivers the best latency**: ~95ms TTFT, ~15ms TPOT
    - **ROCm is on par with CPU** for this model — no GPU acceleration benefit over CPU in llama.cpp
    - **Vulkan edge case**: GPU offloading reduces host memory drastically (0.2 GB vs 4.7–7.0 GB)

### VGGT

!!! warning "VGGT is CPU-Bound"
    VGGT shows **marginal GPU benefit** due to its model architecture. The fastest configuration is iGPU-BF16 on Linux at **30.3s per image** (0.066 img/s).

| Config | Latency | Throughput | Note |
|---|---|---|---|
| CPU (any platform) | 32s–327s | 0.006–0.061 img/s | Linux fastest, WSL slowest |
| iGPU BF16 (Linux) | **30.3s** | **0.066 img/s** | Best result |
| iGPU F32 (Linux) | 43.3s–45.1s | 0.044–0.046 img/s | Negligible SDPA benefit |

    iGPU results on Windows/WSL are incomplete — see [VGGT section](result.md#2-vggt) for details.

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
|---|---|---|
| [Methodology](methodology.md) | Hardware specs, software versions, metric definitions |
| [SmolVLM-Instruct](result.md#1-smolvlm-instruct) | PyTorch + llama.cpp results across all backends |
| [VGGT](result.md#2-vggt) | All configs across platforms |
| [Full SmolVLM Family](result.md#3-full-smolvlm-family-all-models) | All models, PyTorch + llama.cpp |

## Next Steps

!!! example "Planned Work"
    1. ✅ ~~Retest llama.cpp SmolVLM-Instruct on Linux (CPU + Vulkan + ROCm)~~
    2. **Test llama.cpp on Windows** (CPU + Vulkan) and **WSL** (CPU + ROCm)
    3. **Evaluate quantization accuracy** on TextVQA and AI2D benchmarks
    4. **Complete VGGT iGPU testing** on Windows/WSL