# Accelerating Robotics Perception on AMD Ryzen AI APU: A Comparative Port and Benchmark of VGGT and SmolVLM Across Three Compute Engines, CPU/IGPU/NPU

## COMP5709 Capstone Project — University of Sydney

## Project Overview

Robotics systems require fast and efficient perception to operate in real-world environments.

This project explores the capabilities and limitations of **AMD Ryzen AI APU** for robotics perception using:

- CPU
- Integrated GPU (iGPU)
- Neural Processing Unit (NPU) ⚠️

!!! warning "NPU Limitations"

    The Ryzen AI NPU has **severe model compatibility constraints** on this platform (Ryzen 7 8845HS, X1 NPU / XDNA 1):
    
    - **VLM (Vision-Language Models): ❌ Not supported** — only Gemma-3-4b-it has experimental VLM support on X2 NPU (Ryzen AI 300 series), none on X1.
    - **LLM: ⚠️ Limited** — only a small subset of models (e.g. SmolLM2-135M, Qwen-2.5-14b, Phi-4-mini, GPT-OSS-20b) are officially supported. SmolVLM and VGGT are **not** in the supported list.
    - See [AMD Ryzen AI Release Notes](https://ryzenai.docs.amd.com/en/latest/relnotes.html) for the full compatibility matrix, and the [Architecture Overview](General_Advice/Architecture.md) for framework selection guidance.

We focus on two models:

- VGGT
- SmolVLM

## Key Findings

Deploying modern AI perception models on edge hardware is not just about the hardware itself — the software stack and driver support matter enormously. This project benchmarks two representative models across the CPU and iGPU of an AMD Ryzen 7 8845HS APU (Radeon 780M, gfx1103):

- **SmolVLM** — a lightweight vision-language model for multimodal scene understanding
- **VGGT** — a vision-only transformer for 3D geometry estimation

### For Vision-Language Models (SmolVLM)

The clear recommendation is **llama.cpp with Q8_0 quantisation and the Vulkan backend**. This achieves ~93 ms time-to-first-token — roughly **75× faster** than running through PyTorch on the same GPU — while drawing only ~33 W of package power (roughly half of CPU inference). Q8_0 uses 3.9 GB of memory with no accuracy loss from the FP16 baseline. A lower-memory option (Q4_K_M, ~96 ms, 3.2 GB) exists for memory-constrained scenarios, and FP16 (~117 ms, 5.4 GB) for maximum accuracy. The older SmolVLM model is replaced by **SmolVLM2-2.2B-Instruct**, which needs no workaround to run and achieves the same speed, making it the preferred choice for production use.

### For 3D Vision Models (VGGT)

Results are heavily platform-dependent. On **WSL2**, the Windows GPU driver delivers roughly **20× faster** inference than on native Linux (~1.6 s vs. ~30 s per image). The culprit is incomplete convolutional operator support in the community ROCm build for gfx1103 on Linux, not the model itself. On WSL, VGGT becomes viable for near-real-time 3D reconstruction; on Linux, CPU inference is almost as fast as GPU (~36 s vs. ~30 s) and requires far less setup.

### Important Caveats

- **NPU (XDNA 1)** on this device cannot run transformer-based models — a documented hardware limitation.
- The **ROCm backend** in llama.cpp offers no GPU offloading benefit — it draws the same power as CPU inference (~48-52W) while being up to 2.2× slower for FP16. Moreover, attempting to use ROCm can make the system **unstable**, sometimes causing power to spike to ~105W (2× normal), and should be avoided.
- Many quantisation and compilation tools (AMD Quark, IREE, ONNX Runtime with MIGraphX) were evaluated but could not be deployed on this hardware configuration.

## Updates After Report

- Changed image colors to rainbow palette, aligning with the school's recommended style
- Fixed power consumption and memory test offset issues in llama.cpp

## Next Steps

- [ ] **Documentation — General acceleration advice:** Expand the documentation to provide actionable guidance on selecting the right compute engine, quantization level, and deployment path for common robotics perception workloads on AMD Ryzen AI APUs.
- [ ] **Upstream contribution — llama.cpp SmolVLM-Instruct inference issues:** Investigate and contribute fixes upstream for any inference regressions or model compatibility issues encountered when running SmolVLM-Instruct through llama.cpp.
- [ ] **SmolVLM llama.cpp profiling on Windows and WSL:** Conduct detailed profiling of SmolVLM via llama.cpp across both native Windows and WSL2 environments to characterise throughput, latency, and power draw, building on the WSL2 vs. Linux observations from this project.
- [ ] **Quantized model accuracy testing:** Perform systematic accuracy evaluation of the quantized SmolVLM variants (e.g., FP16, Q4_K_M) against the full-precision baseline using relevant perception benchmarks.
- [ ] **Manual MLIR-AIE deployment:** Explore the manual deployment workflow using MLIR-AIE for models mapped to the NPU, as an alternative path for achieving NPU acceleration where standard runtimes fall short.
    