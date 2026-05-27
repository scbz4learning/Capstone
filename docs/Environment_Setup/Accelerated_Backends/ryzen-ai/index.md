# Ryzen AI

Ryzen AI is the official software stack for NPU (Neural Processing Unit) acceleration on AMD hardware. While the installation process has matured significantly, currently it is only recommended for users who **specifically require NPU functionality**. For a broader comparison, see [NPU Inference](../../../Architecture/1_npu-inference.md).

!!! tip "Official Documentation"
    For detailed installation steps and the most up-to-date hardware/software requirements, we highly recommend following the official guides:
    
    - [Ryzen AI Installation on Windows](https://ryzenai.docs.amd.com/en/latest/inst.html)
    - [Ryzen AI Installation on Linux](https://ryzenai.docs.amd.com/en/latest/linux.html)

## Usage Considerations

Based on our empirical testing, the following trade-offs should be considered:

### Linux: GPU-NPU Coexistence Issues
On Linux, APUs currently cannot utilize **MiGraphX**. This has significant implications:
- Unless you are using an APU with the **X2 architecture** (Strix Point/Halo) and require full Ryzen AI model support or advanced NPU features, the stack is limiting.
- Parts of an AI model that are unsupported by the NPU **cannot fall back to the GPU** within the same Python virtual environment. This forces a choice between NPU acceleration or GPU acceleration, but not both simultaneously in a unified environment.

### Windows: Performance Overhead
On Windows, GPUs can be accessed via **DirectML (DML)** within the Ryzen AI stack. However, our benchmarks show that:
- DML performance is considerably weaker than Linux-based **ROCm** (even via WSL).
- It also lags behind **Vulkan** and native **ROCm** implementations.
- If your workload is primarily GPU-heavy, the overhead of the Ryzen AI environment may not be justified.

## Environment Compatibility

!!! warning "ROCm and Ryzen AI cannot coexist"
    ROCm and Ryzen AI **cannot be installed in the same virtual environment**. They are separate software stacks with conflicting dependencies.

## NPU Specifics

!!! note "X1 NPU on this device"
    The XDNA 1 (X1) NPU on this experimental device is **difficult to support** with current tooling and version 1.7.1. For experimental NPU support and manual configuration, refer to the [Developer Guide](../../../Appendix/developer-guide/index.md).
