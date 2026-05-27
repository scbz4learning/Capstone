# GPU Inference

## Overview

GPU support for model inference is quite comprehensive and well-established in the current ecosystem. Unlike NPU solutions which are still evolving and have limited compatibility, GPU acceleration provides superior cross-platform compatibility, supports a much wider range of application scenarios, and offers stable, predictable performance improvements with strong quality guarantees. This makes GPUs an ideal choice for production deployments requiring reliability and consistent acceleration across diverse hardware and software environments.

## Inference Frameworks

Selecting the right inference framework often involves a fundamental trade-off between **throughput** and **latency**. Based on our evaluation:

- **vLLM and SGLang** are primarily optimized for server-side deployments and high-concurrency scenarios. By implementing **PagedAttention** and advanced memory management, these frameworks significantly increase throughput and reduce VRAM fragmentation, making them the preferred choice for Large-Language-Model-as-a-Service (LLMaaS) where processing many requests simultaneously is the priority.

- **PyTorch and llama.cpp** are often more suitable for robotic control, edge computing, and localized interaction. These frameworks prioritize **low latency** and minimal response time over high throughput. While PyTorch offers unmatched flexibility for research and integration with tools like ROS2, llama.cpp provides a lightweight, dependency-free C++ implementation that excels on resource-constrained hardware such as iGPUs and CPUs.

- **ONNX Runtime** offers significant performance gains and cross-vendor stability for fixed production models. By utilizing static graph optimizations, it ensures robust performance across different hardware like NVIDIA, Intel, and AMD. However, it requires more engineering effort for model adaptation, as complex or newer multi-modal architectures may face operator compatibility issues during the export process.

## Acceleration Backends

To drive GPU acceleration, there are 3 primary options available:

- **ROCm and Vulkan (Recommended)** — The most stable and reliable approaches for GPU-accelerated inference. Both have undergone extensive testing and provide robust support across different hardware platforms and software stacks.

- **DirectML (Windows Only)** — Available on Windows systems as an alternative option. However, it typically exhibits lower acceleration efficiency compared to ROCm and Vulkan, and may not be suitable for performance-critical applications.

## ROCm (Radeon Open Compute)

ROCm is the flagship open-source software stack for GPU-accelerated computing, supporting both **Linux** and **Windows** environments. It provides the necessary drivers, development tools, and APIs to leverage the full power of AMD GPUs for AI inference and high-performance computing.

### Installation Options

Users can choose between two main installation paths depending on their hardware and requirements:

- **Official Drivers (Recommended)**: These are the officially supported releases from AMD, providing the most stable and feature-complete experience for supported hardware.
    - **Requirements**: Detailed system requirements can be found for [Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) and [Windows](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html).
    - **Ecosystem Support**: Official drivers are integrated with a wide range of ML frameworks and libraries. See the [ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) for more details.

- **Community Drivers (TheRock)**: For devices not officially listed in the AMD support matrix, the [TheRock](https://github.com/ROCm/TheRock) project offers community-driven builds with broader hardware compatibility.

!!! warning "Potential Desktop Environment Conflicts"
    On Linux systems, installing ROCm drivers via TheRock may conflict with graphical user interfaces (GUIs). It is highly recommended to use these drivers in a **headless** environment (no desktop environment) to ensure system stability.

### TheRock Installation Methods

TheRock project provides multiple ways to deploy ROCm components as described in their [releases documentation](https://github.com/ROCm/TheRock/blob/main/RELEASES.md):

- **Python Packages (via PyPI)** — **Highly Recommended** 
    - **Stability**: Distributing through PyPI ensures access to stable, verified versions of the stack.
    - **Ease of Use**: Installation is a simple `pip install` process without complex system configuration.
    - **Safe Environment**: Using Python packages avoids "system pollution" by keeping the ROCm stack isolated within virtual environments, preventing version conflicts with other system applications.

!!! note "Multi-arch Installation for Unsupported Devices"
    If your specific hardware is not covered by the "Per-family releases", you should use the **Multi-arch PyTorch Python packages** approach. 
    
    For example, attempting to install for `gfx1150` via a family-specific index might fail:
    ```bash
    # This may return an error for non-existent index
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchaudio torchvision
    ```
    Instead, use the unified multi-arch index with device extras:
    ```bash
    # Correct installation method
    pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
        "torch[device-gfx1150]" "torchvision[device-gfx1150]" torchaudio
    ```

- **Tarballs**: Standalone binary archives that can be extracted into any directory, useful for portable or non-standard installations.
- **Native OS Packages**: Standard `.deb` and `.rpm` packages for deep integration with Linux distributions.


## Vulkan

Vulkan is a modern cross-platform graphics and compute API that provides high-efficiency access to modern GPUs. It serves as a great alternative for GPU acceleration, especially where specialized stacks like ROCm are not available or when broad hardware compatibility is required.

### Support and Installation

- **Windows**: On Windows systems, having the latest official graphics drivers (AMD Adrenalin, NVIDIA Game Ready, or Intel Graphics drivers) is typically sufficient. The Vulkan runtime is included in the driver package.
- **Linux**: On Linux, Vulkan support is provided through drivers like **RADV** (part of Mesa for AMD) or the NVIDIA proprietary driver. 
    - For AMD users, the open-source **RADV** driver included in most modern distributions is highly capable.
    - Installation usually involves installing the Vulkan loader and development headers (e.g., `sudo apt install libvulkan-dev` on Ubuntu).

### llama.cpp Integration

Vulkan is a first-class citizen in the **llama.cpp** ecosystem. It allows models to be offloaded to the GPU using the Vulkan backend, which is particularly useful for integrated GPUs (iGPUs) or older hardware. On Linux, the **RADV** driver is often used to execute Vulkan-based inference with excellent performance.

!!! info "[WIP] Ongoing Testing"
    Vulkan-based inference is currently under active testing. Performance benchmarks and detailed configuration guides for different hardware platforms will be updated soon.

