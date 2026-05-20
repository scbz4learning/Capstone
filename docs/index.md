# Accelerating Robotics Perception on AMD Ryzen AI APU

## COMP5709 Capstone Project — University of Sydney

## Project Overview

Robotics systems require fast and efficient perception to operate in real-world environments.

This project explores how **AMD Ryzen AI APU** can accelerate **Vision-Language Models (VLMs)** using:

- CPU
- Integrated GPU (iGPU)
- Neural Processing Unit (NPU)

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
    - Its NPU is **XDNA 1 (X1) architecture** — NPU adaptation is incomplete, so GPU is the primary acceleration target

## Document Structure

This documentation is organized in two parts:

1. **Architecture & Framework Selection** → [`docs/architecture/`](architecture/)
   - General overview of the AMD software stack (ROCm, Ryzen AI, Vulkan, ONNX Runtime, IREE, ...)
   - Decision guide for selecting hardware, framework, and OS based on your setup

2. **Hands-on Guides (SmolVLM & VGGT)** → [`docs/smolvlm/`](smolvlm/) and [`docs/vggt/`](vggt/)
   - Step-by-step environment setup for Ubuntu and Windows
   - Model introductions and example inference code
   - Verified benchmarks and profiling results
   - Known issues and workarounds  