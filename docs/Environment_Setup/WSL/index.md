# WSL2 Environment Setup

Currently on Windows, we recommend using **WSL2** for ROCm-based inference. Since January 2026, Radeon has had **production-level support** for WSL2, which is significantly more stable and performant than native Windows ROCm.

## Reference Documentation

- [ROCm on WSL2 — official installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/wsl/howto_wsl.html)
- [librocdxg — ROCm DirectX GPU interop library](https://github.com/ROCm/librocdxg/)

## Installation Steps

### 1. Install Adrenalin Driver

Install **AMD Software: Adrenalin Edition 26.2.2 for WSL2** or later.

### 2. Install WSL2

Follow [Windows Subsystem for Linux Documentation](https://learn.microsoft.com/en-us/windows/wsl/) to set up WSL2. Ubuntu 24.04 and 22.04 are the supported distros.

### 3. Install ROCm in WSL (Community Driver)

For this experimental device, use TheRock community driver:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install PyTorch with ROCm nightlies
uv pip install --pre torch torchvision torchaudio \
  --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/
```

### 4. Install librocdxg

```bash
git clone https://github.com/ROCm/librocdxg.git
cd librocdxg

# Set the Windows SDK path (adjust version number if different)
export win_sdk='/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/'

# Build the library
mkdir -p build
cd build
cmake .. -DWIN_SDK="${win_sdk}/shared"
make
sudo make install
```

!!! note "Windows SDK Path"
    The Windows SDK path may vary depending on the version installed. Common locations include:
    - `C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\`
    Ensure you have the necessary permissions to access the Windows SDK directory from WSL.

### 5. Load the AMD ROCDXG Library

```bash
export HSA_ENABLE_DXG_DETECTION=1
```

### 6. Fix Library Path

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

## Verify with PyTorch

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

    # Move GPU result back to CPU for comparison
    result_gpu_cpu = result_gpu.to('cpu')

    # Verify correctness
    if torch.allclose(result_cpu, result_gpu_cpu):
        print("CPU and GPU results match!")
    else:
        print("Results differ!")
```

Expected output:
```
CPU operation took: 0.004181 seconds
GPU operation took: 0.001731 seconds
CPU and GPU results match!
```