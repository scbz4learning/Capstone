# Windows Environment Setup

!!! danger "We do not recommend native Windows ROCm"
    Native ROCm on Windows is **incomplete, unstable, and less performant**. For ROCm-based GPU inference, we strongly recommend using **WSL2** instead — AMD provides production-level support there.

    → See [WSL2 Environment Setup](../WSL/index.md)

## PyTorch with ROCm

### Current Windows Support Status

!!! warning "ROCm support on Windows"
    ROCm support on Windows is currently **incomplete and under active development**. Users should expect potential instability, limited feature sets, and library-specific issues compared to the Linux implementation.

#### Key Resources:
- **GPU Compatibility:** Check [SUPPORTED_GPUS.md](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md) for the latest list of verified hardware.
- **Component Support Status:** Refer to [windows_support.md](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md) for a detailed breakdown of ROCm components.
- **GFX1103 Progress:** Track the latest updates for Radeon 780M and other consumer GPUs in [TheRock Issue #1337](https://github.com/ROCm/TheRock/issues/1337).

#### Support Summary:
- **Supported:** Core math libraries (rocBLAS, rocRAND, rocFFT, rocSOLVER, rocSPARSE), ML libraries (MIOpen, hipDNN), and the AMD-LLVM compiler toolchain are generally functional.
- **Unsupported/Limited:** Profiling tools (rocprofiler-sdk, aqlprofile), communication libraries (RCCL), and media decoding (rocDecode, rocJPEG) are currently unsupported or restricted on Windows. System-level tools like `amdsmi` and `rocr-runtime` are also pending full support.

### Prerequisites

1. Install the latest [Adrenaline driver](https://www.amd.com/en/products/software/adrenalin.html).
2. Read TheRock release guidance (Windows and compatibility notes): https://github.com/ROCm/TheRock/blob/main/RELEASES.md.

### ROCm installation paths

Support three ways to install ROCm:

#### Option 1 (recommended for most users): pip + nightlies

```bash
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ "rocm[libraries,devel]"
```

#### Option 2 (recommended for reproducibility): artifacts download

1. Clone [TheRock repository](https://github.com/ROCm/TheRock)
2. Use the artifact installation helper:
    - `TheRock\build_tools\install_rocm_from_artifacts.py`
3. Supported channels:
    - `dev` (recommended)
    - `nightly`
4. Optional: target a specific GitHub Actions run ID for fixed behavior.

Check more in [Artifact install docs](https://github.com/ROCm/TheRock/blob/main/docs/development/installing_artifacts.md)

#### Option 3 (source build, least recommended)

Source builds require ~100GB disk, tons of hours, should be equivalent to artifacts in code path. Use only if you need custom build control.

##### Prerequisites and Set up Environment

- Confirm prerequisites in [RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
- Reinstall Git with "Use Git and optional Unix tools from the Windows Command Prompt".
- Set these git options:
    - `git config --global core.symlinks true`
    - `git config --global core.longpaths true`
    - `git config --global core.autocrlf true`
- Clone https://github.com/ROCm/TheRock.
- Validate environment using:
    - `.\TheRock\build_tools\validate_windows_install.ps1`

!!! note "VS Build Tools environment note"

    - `x64 Native Tools Command Prompt for VS 2022` may fail PowerShell scripts.
    - `Developer PowerShell for VS 2022` is x86 by default, not sufficient for TheRock builds.
    - Use `scripts\activate_building_tools.ps1` in this repo to map `vsDevCmd` output into PowerShell.
    - Usage: Open PowerShell, copy the script content and run in terminal. Note that `vsDevCmd` path needs adjustment based on actual install.

##### Build flow

- `uv pip install -r requirements.txt`. If venv is not activated, activate it first.
- `uv run python ./build_tools/fetch_sources.py`. (Takes a long time, handles many files).
- `TheRock\build_tools\setup_ccache.py` (Optional).
- Build: `cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx110X-all`

##### Additional docs:

- https://github.com/ROCm/TheRock/blob/main/docs/development/README.md
- https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md

### PyTorch installation

This step usually has no issues. Currently supports Python 3.10-3.12 for Torch 2.9 & 2.10.

Recommended:
```bash
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ torch torchaudio torchvision
```

!!! note "Other Methods"
    This pulls prebuilt wheels that usually have the best compatibility.

    If your device still has compatibility issues, inspect the [Windows PyTorch wheel release workflow](https://github.com/ROCm/TheRock/actions/workflows/release_windows_pytorch_wheels.yml) for artifact options or source build instructions.

### Fixing Known issue: HIP API error 0100 on iGPU mapping

`checkHipErrors() HIP API error = 0100 "no ROCm-capable device is detected"`

#### Observed GPU enumeration issue on Windows

TheRock issues confirm that Windows iGPU + discrete GPU enumeration is still under active work:

- [AMD iGPU is cuda:0 when using ROCm wheels built by TheRock](https://github.com/ROCm/TheRock/issues/3392)
    - AMD developer confirms Windows device enumeration priority is buggy and can skip the iGPU or conflict with dGPU.
    - Standard workaround: set `HIP_VISIBLE_DEVICES=0` explicitly.
- [Confused about current status of support for Radeon 780M (gfx1103) on Windows](https://github.com/ROCm/TheRock/issues/3031)
    - Confirms gfx1103 support in TheRock nightlies is turned on, but some libraries are not yet fully bulletproof.
- [Hangs/fails on Radeon 780M (gfx1103)](https://github.com/ROCm/TheRock/issues/1264)
    - Multiple iGPU users report silent lockups.

#### Workaround (for current session)

To force the HIP runtime to recognize the iGPU, set `HIP_VISIBLE_DEVICES=0`:

**PowerShell:**
```powershell
$env:HIP_VISIBLE_DEVICES = "0"
```

**cmd.exe:**
```cmd
set HIP_VISIBLE_DEVICES=0
```

> To make permanent, add `HIP_VISIBLE_DEVICES` user environment variable value `0` in Windows Settings.

### Final Verification and Testing

```python
import torch
import time

print("ROCm:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Create deterministic tensors
tensor_a_cpu = torch.full((1000, 1000), 2.0, device='cpu')
tensor_b_cpu = torch.full((1000, 1000), 3.0, device='cpu')

# CPU computation
start_time = time.time()
result_cpu = tensor_a_cpu + tensor_b_cpu
cpu_time = time.time() - start_time
print(f"CPU operation took: {cpu_time:.6f} seconds")

# GPU computation
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

## ONNX Runtime

There are two ways to run ONNX Runtime on Windows:

### Method 1: ONNX Runtime-ROCm (via ROCm / MIGraphX EP)

Uses the ROCm or MIGraphX Execution Provider. Note that the **ROCm EP has been deprecated**, and **MIGraphX EP may have issues on this experimental device**.

### Method 2: Ryzen AI (via DirectML)

Uses Ryzen AI software stack with DirectML for inference.

!!! warning "ROCm and Ryzen AI cannot coexist"
    ROCm and Ryzen AI **cannot be installed in the same virtual environment**. They are separate stacks.

### Recommendation

ONNX Runtime on Windows is **not recommended**. In most cases, using **PyTorch with ROCm** or **Vulkan** (via llama.cpp / vLLM) yields better stability and performance.

## NPU

!!! note "X1 NPU on this device"
    The XDNA 1 (X1) NPU on this experimental device is **difficult to support** with current tooling. For experimental NPU support, refer to the [Developer Guide](../../appendix/developer-guide.md).