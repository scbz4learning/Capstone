# NPU Support on AMD Ryzen AI

NPU support for XDNA 1 (X1) architecture is currently in **early development stage**. This guide covers the key projects and resources for developers looking to work with the AMD NPU.

## Key Projects

### 1. ROCm / MLIR

The [ROCm](https://rocm.docs.amd.com) project includes MLIR-based compiler infrastructure for AMD GPUs. While primarily targeting RDNA/CDNA GPUs, ongoing work extends MLIR support toward AI Engine (NPU) code generation.

- Repository: [https://github.com/ROCm/ROCm](https://github.com/ROCm/ROCm)
- MLIR documentation: [https://rocm.docs.amd.com/projects/MLIR](https://rocm.docs.amd.com/projects/MLIR)

### 2. AMD AI Engine (AIE)

The AMD AI Engine is the compute fabric inside XDNA NPUs. The AIE toolchain provides low-level access to the NPU for custom kernel development.

- AIE documentation: [https://ryzenai.docs.amd.com](https://ryzenai.docs.amd.com)
- AIE MLIR dialect for custom operator compilation
- Used internally by Ryzen AI software stack and Vitis AI

### 3. IREE

[IREE](https://iree.dev) (Intermediate Representation Execution Environment) is an MLIR-based compiler and runtime that can target multiple hardware backends, including AMD NPU.

- Repository: [https://github.com/openxla/iree](https://github.com/openxla/iree)
- AMD backend status: CPU and GPU support are stable; NPU support is experimental
- Supports both Linux and Windows
- Workflow: Model → IREE → MLIR → target-specific code (CPU / GPU / NPU)

## Current Limitations (X1 NPU)

- No official high-level framework (PyTorch / ONNX Runtime) support for X1 NPU on this platform
- X1 NPU adaptation in Ryzen AI software stack is incomplete
- Recommended path for NPU experimentation: IREE (experimental) or Vitis AI (Windows only)

## Relevant Links

- [Ryzen AI Software](https://ryzenai.docs.amd.com)
- [Vitis AI](https://github.com/Xilinx/Vitis-AI)
- [IREE AMD Backend](https://iree.dev/guides/deploying-on-amd/)
- [ROCm TheRock](https://github.com/ROCm/TheRock)