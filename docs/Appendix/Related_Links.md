# Related Links

## 1. Models

### 3D Vision

- https://github.com/facebookresearch/vggt — [CVPR 2025 Best Paper] VGGT: Visual Geometry Grounded Transformer (Meta AI)
- https://github.com/bsun0524/vggt — Your fork of VGGT (forked from facebookresearch)
- https://github.com/akretz/vggt-onnx — ONNX model exports of VGGT
- https://github.com/akretz/vggt-onnx/issues/3 — Issue tracking: matching PyTorch performance in ONNX export
- https://docs.google.com/document/d/1nneDNJIQsQIcZE25f1hMXWhfY_tUfYP4IVbSVQTJiuQ/edit — WSL2 VGGT 模型性能分析 (Performance Analysis)
- https://vgg-t.github.io/ — VGGT official project page
- https://arxiv.org/abs/2503.11651 — VGGT arXiv paper

### Multimodal LLM

- https://github.com/ggml-org/llama.cpp/blob/master/conversion/smolvlm.py — SmolVLM model conversion script
- https://github.com/ggml-org/llama.cpp/issues/10877 — Feature request: SmolVLM support
- https://github.com/InternLM/lmdeploy/issues/3089 — SmolVLM multimodal support in lmdeploy


## 2. Inference Frameworks

### llama.cpp

- https://github.com/ggml-org/llama.cpp — LLM inference in C/C++ (114k stars, HIP/Vulkan/Metal backends)
- https://github.com/ggml-org/llama.cpp/pull/13050 — PR: SmolVLM (v1 & v2) multimodal support
- https://github.com/ggml-org/llama.cpp/pull/12898 — PR: vision support in llama-server via libmtmd
- https://github.com/ggml-org/llama.cpp/issues/17871 — Issue: SmolVLM jinja template parsing bug
- https://github.com/ggml-org/llama.cpp/issues/15971 — Issue: SmolVLM2 core dump on run
- https://github.com/ggml-org/llama.cpp/issues/21634 — Issue: SmolVLM tokenize prompt failure
- https://github.com/ggml-org/llama.cpp/discussions/16938 — Guide: new WebUI of llama.cpp

### vLLM
- https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#amd-rocm — GPU installation (vLLM)
- https://docs.vllm.ai/en/latest/configuration/conserving_memory/#multi-modal-input-limits — Conserving memory (multi-modal)
- https://rocm.docs.amd.com/en/7.13.0-preview/ai-inference/vllm.html — vLLM on ROCm 7.13.0 preview
- https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html — vLLM V1 performance optimization

### SGLang
- https://docs.sglang.io/docs/hardware-platforms/amd_gpu — SGLang AMD GPU support

### AMD-SHARK
- https://github.com/nod-ai/amd-shark-ai — AMD-SHARK inference modeling and serving

### lmdeploy
- https://github.com/InternLM/lmdeploy/issues/3089 — [Feature] Add support for SmolVLM multimodal models


## 3. Acceleration Backends

### ROCm

- https://github.com/ROCm/TheRock — The HIP Environment and ROCm Kit (lightweight HIP/ROCm build system)
- https://github.com/ROCm/TheRock/blob/main/README.md — TheRock README
- https://github.com/ROCm/TheRock/blob/main/RELEASES.md — Installation instructions
- https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md — Supported GPUs
- https://github.com/ROCm/TheRock/issues/3618 — gfx1103/780M LLM issue on Ubuntu 24.04
- https://github.com/ROCm/TheRock/issues/3044 — [gfx1103] HSA_ERROR_INVALID_ISA in FP16
- https://github.com/ROCm/TheRock/issues/4954 — gfx1151 MIOpen conv solver failure
- https://github.com/ROCm/TheRock/issues/4743 — gfx110x nightly CI failures
- https://github.com/ROCm/AMDMIGraphX — AMD's graph inference engine for ML
- https://github.com/ROCm/librocdxg/ — ROCDXG: AMD ROCDXG project (DirectX on Linux/WSL)
- https://github.com/agrocylo/bitsandbytes-rocm — 8-bit CUDA functions ported to HIP for AMD GPUs
- https://github.com/Dao-AILab/flash-attention — Fast attention with ROCm support
- https://github.com/Dao-AILab/flash-attention/issues/965 — ROCm support discussion
- https://github.com/ROCm/rocm-libraries/pull/5763 — [gfx1103] Disable Winograd Fury workaround
- https://rocm.docs.amd.com/en/latest/ — ROCm documentation home
- https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html — ROCm compatibility matrix
- https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/index.html — Use ROCm for AI inference
- https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/index.html — ROCm AI inference optimization
- https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch.html — GPU architecture documentation

### Ryzen AI (NPU)

- https://ryzenai.docs.amd.com/en/latest/ — Ryzen AI Software 1.7.1 documentation
- https://ryzenai.docs.amd.com/en/latest/llm/overview.html — LLM Deployment Overview
- https://ryzenai.docs.amd.com/en/latest/indu.html — Installation Instructions
- https://ryzenai.docs.amd.com/en/latest/model_quantization.html — Model Quantization
- https://ryzenai.docs.amd.com/en/latest/ops_support.html — Supported Operators
- https://ryzenai.docs.amd.com/en/latest/xrt_smi.html — NPU Management Interface
- https://ryzenai.docs.amd.com/en/latest/examples.html — Examples, Demos, Tutorials

### Vulkan

- https://iree.dev/guides/deployment-configurations/gpu-vulkan/ — IREE GPU - Vulkan deployment config
- https://iree.dev/developers/performance/profiling-gpu-vulkan/ — Profiling GPUs using Vulkan (IREE)
- https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-VULKAN.html — Vulkan driver support
- https://github.com/ggml-org/llama.cpp/releases/tag/b9388 — llama.cpp Vulkan AMD UMA fix (b9388)
- https://github.com/ggml-org/llama.cpp/pull/22455 — Avoid preferring transfer queue on AMD UMA devices

### DirectML / Windows

- https://ryzenai.docs.amd.com/en/latest/gpu/ryzenai_gpu.html — DirectML Flow (Ryzen AI)
- https://ryzenai.docs.amd.com/en/latest/winml/winml_overview.html — Windows ML Overview
- https://ryzenai.docs.amd.com/en/latest/winml/winml_example.html — Windows ML Example
- https://rocm.docs.amd.com/projects/install-on-windows/en/latest/ — HIP SDK for Windows


## 4. OS

### Linux

- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html — Quick start installation guide
- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/detailed-install.html — Detailed install
- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html — PyTorch on ROCm installation
- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/dgl-install.html — DGL on ROCm installation
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-radeon.html — Install Radeon software for Linux
- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/prerequisites.html — Installation prerequisites
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/howto_native_linux.html — Linux How to guide (Ryzen)
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/advanced/advancedrad/usecases.html — Radeon Usecases

### WSL

- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html — WSL How to guide (Radeon)
- https://github.com/ROCm/TheRock/blob/main/docs/development/wsl_rocdxg.md — WSL ROCDXG development
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.1.3/docs/install/wsl/install-radeon.html — Install Radeon software for WSL

### Windows (Native)

- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html — Windows PyTorch install
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/howto_windows.html — How to guide - Windows


## 5. Hardware

### AMD GPU Compatibility

- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/compatibility.html — Radeon compatibility matrices
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html — Linux support matrices (Radeon)
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html — Windows support matrices (Radeon)
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/prerequisites/prerequisitesrad.html — Prerequisites for Radeon desktop GPUs
- https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html — GPU hardware specifications
- https://rocm.docs.amd.com/en/latest/reference/precision-support.html — Data types and precision support
- https://www.amd.com/en/products/software/adrenalin.html — AMD Software: Adrenalin Edition
- https://www.amd.com/en/developer/uprof.html — AMD μProf

### CPU / APU

- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/prerequisites/prerequisitesryz.html — Prerequisites for Ryzen APUs
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/compatibility.html — Ryzen compatibility matrices
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html — Install Ryzen Software for Linux


## Uncategorized

### Quantization & Optimization

- https://quark.docs.amd.com/latest/ — AMD Quark documentation
- https://quark.docs.amd.com/latest/pytorch/basic_usage_pytorch.html — AMD Quark for PyTorch
- https://quark.docs.amd.com/latest/tutorials/onnx/accuracy_improvement/gptq/onnx_gptq_tutorial.html — Quark ONNX GPTQ Tutorial
- https://github.com/microsoft/olive — Olive: ML model finetuning, conversion, quantization, and optimization

### Inference Optimization

- https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-quantization.html — Model quantization techniques
- https://rocm.docs.amd.com/en/7.13.0-preview/ai-inference/conceptual/optimize-triton-kernels.html — Optimizing Triton kernels
- https://rocm.docs.amd.com/en/7.13.0-preview/ai-inference/conceptual/model-quantization.html — Model quantization
- https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html — AMD Instinct MI300X workload optimization
- https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/profiling-and-debugging.html — Profiling and debugging
- https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html — Model acceleration libraries
- https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/hugging-face-models.html — Running HF models on ROCm
- https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/llm-inference-frameworks.html — LLM inference frameworks

### Issues Related

- https://github.com/ROCm/TheRock/issues/136 — Install AMD GPU driver on clean machine
- https://github.com/ROCm/TheRock/issues/1937 — gfx1151 torch.compile system freeze
- https://github.com/ROCm/TheRock/issues/2139 — Retrieve tarballs from CloudFront
- https://github.com/ROCm/TheRock/issues/2425 — ROCm Docker support migration
- https://github.com/ROCm/TheRock/issues/247 — JAX support
- https://github.com/ROCm/TheRock/issues/3323 — Multi-arch packaging with Kpack
- https://github.com/ROCm/TheRock/issues/3905 — Windows gfx110x smoketests failing
- https://github.com/ROCm/TheRock/issues/4105 — Unit test coverage
- https://github.com/ROCm/TheRock/issues/4214 — SDXL VAE HIP error gfx1150
- https://github.com/ROCm/TheRock/issues/4809 — Skip unsupported subprojects
- https://github.com/ROCm/TheRock/issues/5021 — PyTorch segfault gfx1151
- https://github.com/ROCm/TheRock/issues/5172 — Windows ROCm wheel cancellation
- https://github.com/ROCm/TheRock/issues/5178 — libtorch_hip.so linking failure
- https://github.com/ROCm/TheRock/issues/5237 — Improve generate_s3_index.py
- https://github.com/ROCm/TheRock/issues/5355 — gfx950 pytorch libnuma error
- https://github.com/ROCm/TheRock/issues/5403 — Install amdrocm-core-sdk fails
- https://github.com/ROCm/TheRock/issues/157 — Windows gfx1151 ROCm wheel install cancelling
- https://github.com/ROCm/AMDMIGraphX/issues/4146 — Compile error with different clang++
- https://github.com/ROCm/AMDMIGraphX/issues/4272 — Python 3.10 support issue
- https://github.com/ggml-org/llama.cpp/issues/965 — Does flash-attention support ROCm?
- https://github.com/ggml-org/llama.cpp/issues/10877 — Feature request: SmolVLM support
- https://github.com/InternLM/lmdeploy/issues/3089 — SmolVLM multimodal support in lmdeploy
- https://github.com/ROCm/AMDMIGraphX/releases/tag/rocm-7.2.3 — MIGraphX rocm-7.2.3 release
- https://github.com/akretz/vggt-onnx/actions — CI workflows
