# ROCm

## Linux (Ubuntu)

### Install `uv` and Create Virtual Environment

```bash
sudo apt update
sudo apt install curl -y
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv venv
source venv/bin/activate
```

### Install Pre-built Packages (Recommended)

```bash
uv pip install --pre torch torchvision torchaudio \
  --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/
```

### Build Tools (for Source Build)

```bash
sudo apt install gfortran git ninja-build cmake g++ pkg-config xxd patchelf automake libtool python3-venv python3-dev libegl1-mesa-dev texinfo bison flex
```

Source build reference: [TheRock repository](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#manual-tarball-extraction)

### Test ROCm with PyTorch

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

### Legacy Note

Before Dec 2025, ROCm required `export HSA_OVERRIDE_GFX_VERSION=11.0.0`. This is no longer necessary.

---

## Windows

!!! danger "We do not recommend native Windows ROCm"
    Native ROCm on Windows is **incomplete, unstable, and less performant**. For ROCm-based GPU inference, we strongly recommend using **WSL2** instead.

→ See [WSL2 section below](#wsl2)

### Prerequisites

1. Install latest [Adrenalin driver](../../Drivers/index.md)
2. Read [TheRock release guidance](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)

### ROCm Installation Paths

#### Option 1 (recommended): pip + nightlies

```bash
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ "rocm[libraries,devel]"
```

#### Option 2 (artifacts download)

1. Clone [TheRock repository](https://github.com/ROCm/TheRock)
2. Run: `TheRock\build_tools\install_rocm_from_artifacts.py`
3. Channels: `dev` (recommended) or `nightly`

See: [Artifact install docs](https://github.com/ROCm/TheRock/blob/main/docs/development/installing_artifacts.md)

#### Option 3 (source build, least recommended)

Requires ~100GB disk. Use only if you need custom build control.

```bash
# Prerequisites
git config --global core.symlinks true
git config --global core.longpaths true
git config --global core.autocrlf true
```

See: [TheRock Windows build docs](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md)

### PyTorch Installation

```bash
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ torch torchaudio torchvision
```

Supports Python 3.10-3.12 for Torch 2.9 & 2.10.

### Known Issue: HIP API error 0100

`checkHipErrors() HIP API error = 0100 "no ROCm-capable device is detected"`

Workaround: `$env:HIP_VISIBLE_DEVICES = "0"`

Related issues:

- [AMD iGPU is cuda:0](https://github.com/ROCm/TheRock/issues/3392)
- [gfx1103 support on Windows](https://github.com/ROCm/TheRock/issues/3031)
- [Hangs on Radeon 780M](https://github.com/ROCm/TheRock/issues/1264)

---

## WSL2

Since January 2026, Radeon has **production-level support** for WSL2.

### Reference Documentation

- [ROCm on WSL2 installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/wsl/howto_wsl.html)
- [librocdxg — ROCm DirectX GPU interop](https://github.com/ROCm/librocdxg/)

### Installation Steps

#### 1. Install WSL2

Follow [WSL2 documentation](https://learn.microsoft.com/en-us/windows/wsl/). Ubuntu 24.04 and 22.04 are supported.

#### 2. Install ROCm in WSL (Community Driver)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install --pre torch torchvision torchaudio \
  --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/
```

#### 3. Install librocdxg

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

#### 4. Environment Variables

```bash
export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

---

## ONNX Runtime

### Linux

```python
import onnxruntime as ort
ort.get_available_providers()
```

Expected: `MIGraphXExecutionProvider` and `CPUExecutionProvider`.

See [Install ONNX Runtime for Radeon GPUs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.1.3/docs/install/native_linux/install-onnx.html).

### Windows

Two methods:
1. **ROCm / MIGraphX EP** — ROCm EP is deprecated; MIGraphX EP may have issues.
2. **Ryzen AI (DirectML)** — see [ryzen-ai](../ryzen-ai/index.md).

!!! warning "ROCm and Ryzen AI cannot coexist in the same venv"
