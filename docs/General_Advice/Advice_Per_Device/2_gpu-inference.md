# GPU Inference

Before settling on CPU, first determine whether your device supports GPU
acceleration through the decision flowchart below.

## Decision Flowchart

```
Official driver
[Radeon compatibility matrices](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/compatibility.html)
[Ryzen compatibility matrices](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/compatibility.html)
├── Yes → Use official driver. Choose any inference framework that supports ROCm or Vulkan; results may vary by model
└── No → Do you have access to WSL2 (Windows Subsystem for Linux)?
    ├── Yes → Use WSL2 with librocdxg (ROCm DXG bridge)
    │   WSL2 routes GPU compute through the Windows AMD driver,
    │   which has production-level kernel support (especially for Conv ops).
    │   See: https://github.com/ROCm/librocdxg
    │   Usually faster than native Linux for Conv-heavy models.
    │
    └── No → Check if any inference framework supports Vulkan for the model
        ├── Yes → Use Vulkan to drive GPU inference
        └── No → Try TheRock (https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md) community build for GPU inference
            Usually fine, even if slower — but note: TheRock's MIOpen
            (convolution kernel library) is in early preview and may lack
            optimized Conv kernels, causing significant slowdowns for
            Conv-heavy models (e.g. VGGT sees ~19× slowdown vs WSL).
            If missing or problematic — known issues on GitHub, or operators that are truly special → fall back to CPU
```

---

## Driver Installation

### Ubuntu / Linux

#### Install AMDGPU Driver

The Linux kernel usually includes the `amdgpu` driver by default, so a separate driver installation is generally not required.

If PyTorch does not run correctly or GPU detection fails, some users have reported that uninstalling `amdgpu-dkms` can help. See the discussion in [ROCm/TheRock issue #3618](https://github.com/ROCm/TheRock/issues/3618#issuecomment-4281691473) and the official AMD quick-start guide at https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html for the recommended recovery steps.

### Windows

Install the latest [Adrenalin driver](https://www.amd.com/en/products/software/adrenalin.html).

#### WSL2

Install **AMD Software: Adrenalin Edition 26.2.2 for WSL2** or later to ensure proper GPU passthrough to the Linux subsystem.


---

## Accelerated Backends

### 1. ROCm
#### Compatibility

[Radeon compatibility matrices](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/compatibility.html)
[Ryzen compatibility matrices](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/compatibility.html)

#### Installation

Follow [ROCm on Radeon and Ryzen](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html).

#### Key Capabilities

[ROCm Key Capabilities](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html#rocm-key-capabilities)

---

### 2. Vulkan

Vulkan is a modern cross-platform graphics and compute API that provides
high-efficiency access to modern GPUs.

#### Installation

See the **Driver Installation** section above.

- **Windows**: Ensure you have the latest AMD Adrenalin, NVIDIA Game Ready, or Intel Graphics driver. The Vulkan runtime is included.
- **Linux**: On modern distributions such as Ubuntu 24.04 LTS, the Vulkan driver should already present. 

!!! tip "Missing Vulkan?"

    A community advice is as below (**NOT TESTED by author**):

    > If missing, install:
    >
    > ```bash
    > sudo apt install mesa-vulkan-drivers
    > ```
    > 
    > This covers both the loader and the open-source RADV driver.

---

### 3. TheRock (Community ROCm)

!!! warning "Check support status before using TheRock"
    Community ROCm builds are not as stable or complete as official ROCm releases. Before proceeding, verify your GPU's support status:
    - [SUPPORTED_GPUS.md](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
    - [Windows support status](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md)

!!! info "Where to get help"
    Community ROCm is less stable and you may encounter various issues. For support, check:
    - [Known Issues](../../Appendix/known-issues.md)
    - [Environment setup guide](https://github.com/ROCm/TheRock/blob/main/docs/environment_setup_guide.md)
    - [FAQ](https://github.com/ROCm/TheRock/blob/main/docs/faq.md)
    - [GitHub Issues](https://github.com/ROCm/TheRock/issues)

!!! example "Example hardware: Radeon 780M (gfx1103)"
    The installation steps below use Radeon 780M (`gfx1103`) as the example target.

#### Linux (Ubuntu)

##### Install `uv` and Create Virtual Environment

```bash
sudo apt update
sudo apt install curl -y
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv venv
source venv/bin/activate
```

##### Install Pre-built Packages (Recommended)

!!! note "Multi-arch Installation for Unsupported Devices"
    If your specific hardware is not covered by the "Per-family releases", you should use the multi-arch PyTorch Python packages approach.

    For example, attempting to install for `gfx1150` via a family-specific index might fail:
    ```bash
    # This may return an error for non-existent index
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchvision torchvision
    ```
    Instead, use the unified multi-arch index with device extras:
    ```bash
    # Correct installation method
    pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
        "torch[device-gfx1150]" "torchvision[device-gfx1150]" torchaudio
    ```

```bash
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ torch torchvision torchaudio
```


##### Build Tools (for Source Build)

```bash
sudo apt install gfortran git ninja-build cmake g++ pkg-config xxd patchelf automake libtool python3-venv python3-dev libegl1-mesa-dev texinfo bison flex
```

Source build reference: [TheRock repository](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#manual-tarball-extraction)

##### Test ROCm with PyTorch

```python
import torch
import time

print("ROCm:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

tensor_a_cpu = torch.full((1000, 1000), 2.0, device='cpu')
tensor_b_cpu = torch.full((1000, 1000), 3.0, device='cpu')

start_time = time.time()
result_cpu = tensor_a_cpu + tensor_b_cpu
cpu_time = time.time() - start_time
print(f"CPU operation took: {cpu_time:.6f} seconds")

if torch.cuda.is_available():
    tensor_a_gpu = tensor_a_cpu.to('cuda')
    tensor_b_gpu = tensor_b_cpu.to('cuda')

    torch.cuda.synchronize()
    start_time = time.time()
    result_gpu = tensor_a_gpu + tensor_b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    print(f"GPU operation took: {gpu_time:.6f} seconds")

    result_gpu_cpu = result_gpu.to('cpu')

    if torch.allclose(result_cpu, result_gpu_cpu):
        print("CPU and GPU results match!")
    else:
        print("Results differ!")
```

Sample output:
```
CPU operation took: 0.004181 seconds
GPU operation took: 0.001731 seconds
CPU and GPU results match!
```

##### Legacy Note

Before Dec 2025, ROCm required `export HSA_OVERRIDE_GFX_VERSION=11.0.0`. This is no longer necessary.

#### Windows

##### Prerequisites

1. Install latest [Adrenalin driver](https://www.amd.com/en/products/software/adrenalin.html)
2. Read [TheRock release guidance](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)

##### ROCm Installation Paths

###### Option 1 (recommended): pip + nightlies

```bash
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ "rocm[libraries,devel]"
```

###### Option 2 (artifacts download)

1. Clone [TheRock repository](https://github.com/ROCm/TheRock)
2. Run: `TheRock\build_tools\install_rocm_from_artifacts.py`
3. Channels: `dev` (recommended) or `nightly`

See: [Artifact install docs](https://github.com/ROCm/TheRock/blob/main/docs/development/installing_artifacts.md)

###### Option 3 (source build, least recommended)

Requires ~100GB disk. Use only if you need custom build control.

```bash
# Prerequisites
git config --global core.symlinks true
git config --global core.longpaths true
git config --global core.autocrlf true
```

See: [TheRock Windows build docs](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md)

##### PyTorch Installation

!!! note "Multi-arch Installation for Unsupported Devices"
    If your specific hardware is not covered by the "Per-family releases", you should use the multi-arch PyTorch Python packages approach.

    For example, attempting to install for `gfx1150` via a family-specific index might fail:
    ```bash
    # This may return an error for non-existent index
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchvision torchvision
    ```
    Instead, use the unified multi-arch index with device extras:
    ```bash
    # Correct installation method
    pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
        "torch[device-gfx1150]" "torchvision[device-gfx1150]" torchaudio
    ```

```bash
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ torch torchaudio torchvision
```

Supports Python 3.10-3.12 for Torch 2.9 & 2.10.

##### Known Issues

See the dedicated known-issues appendix: [Known Issues](../../Appendix/known-issues.md)

#### WSL2

Since January 2026, Radeon has **production-level support** for WSL2. As of [librocdxg v1.2.0](https://github.com/ROCm/librocdxg/releases/tag/v1.2.0) (May 2026), `gfx1103` and `gfx1152` are officially supported in the supported device list.

!!! tip "Why WSL2 can be faster than native Linux"
    WSL2 uses [librocdxg](https://github.com/ROCm/librocdxg) to bridge ROCm/HIP compute calls to the **production-grade Windows AMD driver** via DXCore/DXG. In contrast, native Linux on unsupported GPUs relies on TheRock's MIOpen, which is in early preview and may have incomplete convolution kernel databases. This means Conv-heavy models (e.g. VGGT) can be ~19× faster on WSL2, while Transformer-heavy models (e.g. SmolVLM) show no difference. See the detailed [GPU Driver Stack Analysis](../../Experience_For_Developers/Per_Device/GPU.md) for the full technical breakdown.

##### Reference Documentation

- [ROCm on WSL2 installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/wsl/howto_wsl.html)
- [librocdxg — ROCm DirectX GPU interop](https://github.com/ROCm/librocdxg/)

##### Installation Steps

###### 1. Install WSL2

Follow [WSL2 documentation](https://learn.microsoft.com/en-us/windows/wsl/). Ubuntu 24.04 and 22.04 are supported.

###### 2. Install ROCm in WSL (Community Driver)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install --pre torch torchvision torchaudio \
  --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/
```

###### 3. Install librocdxg

```bash
git clone https://github.com/ROCm/librocdxg.git
cd librocdxg

export win_sdk='/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/'

mkdir -p build
cd build
cmake .. -DWIN_SDK="${win_sdk}/shared"
make
sudo make install
```

###### 4. Environment Variables

```bash
export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

---

## CPU Fallback

If none of the above paths are viable for your hardware or model, fall back
to CPU inference.

- **llama.cpp** — low-latency CPU inference with optimized kernels
- **ONNX Runtime** — CPU execution provider with graph optimizations
- **PyTorch** — flexible CPU execution; consider `torch.compile` and quantization

---

## Summary

| Path | Readiness | When to Use |
|------|-----------|-------------|
| ROCm (Official) | Production | Device in AMD compatibility tables |
| Vulkan | Production | No ROCm, but Vulkan-capable driver available |
| WSL2 (librocdxg) | Production (Windows driver) | Device not officially supported; Windows host available. Routes ROCm through production AMD Windows driver via DXG bridge — often significantly faster than TheRock for Conv-heavy models |
| TheRock | Community | Device not officially supported; headless Linux only; no Windows host available |
| CPU | Baseline | No GPU acceleration path available |

The recommended priority is: **ROCm → Vulkan → WSL2 (librocdxg) → TheRock → CPU**.
