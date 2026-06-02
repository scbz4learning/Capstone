# Accelerating Robotics Perception on AMD Ryzen AI APU

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

## Experimental Platform

The experiments are conducted on a compact edge device:

**Beelink SER Mini PC**

- AMD Ryzen 7 8845HS  
- 32 GB RAM  
- 1 TB SSD

!!! note "Experimental Device Constraints"
    - This device has **no official ROCm support**; it relies on **community ROCm support** via [TheRock](https://github.com/ROCm/TheRock)

## Document Structure

This documentation is organized as follows:

1. **Architecture & Framework Selection** → [`docs/architecture/`](architecture/)    
    - General overview of the AMD software stack (ROCm, Ryzen AI, Vulkan, ONNX Runtime, IREE, ...)  
    - Decision guide for selecting hardware, framework, and OS based on your setup  

2. **Hands-on Guides (SmolVLM & VGGT)** → [`docs/smolvlm/`](smolvlm/) and [`docs/vggt/`](vggt/)    
    - Step-by-step environment setup for Ubuntu and Windows  
    - Model introductions and example inference code  
    - Known issues and workarounds (NPU VLM unsupported, limited LLM support)

3. **Profiling & Benchmarking** → [`docs/profiling/`](profiling/)
    - Comprehensive performance profiling results across platforms (Windows, WSL, Ubuntu)
    - Detailed latency, throughput, power consumption, and memory usage metrics
    - Platform comparison and recommendations
    - Performance optimization insights    