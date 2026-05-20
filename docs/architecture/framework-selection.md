# Framework Selection Guide

## Decision Flow

```
Start
  │
  ├── Which hardware do you have?
  │   ├── X2 NPU (e.g. Ryzen AI 300 series)
  │   │   └── Prioritize NPU or NPU+GPU hybrid inference (fastest path)
  │   │
  │   └── X1 NPU or no NPU
  │       └── Prioritize GPU — X1 NPU adaptation is still immature
  │
  ├── Which framework should you use?
  │   ├── Check AMD's Hugging Face organization for model support
  │   │   ├── Model officially supported → Use Ryzen AI toolchain (easiest path to AMD hardware)
  │   │   └── Not supported
  │   │       ├── X2 NPU available → Use Ryzen AI OGA framework
  │   │       │   └── Quantize with Microsoft Olive or AMD Quark
  │   │       │       └── Enables NPU or NPU+GPU hybrid inference
  │   │       └── X1 NPU or no NPU → Use GPU framework
  │   │           ├── ROCm (official device → ROCm supported list;
  │   │           │    unsupported device → TheRock community support, stabilizing)
  │   │           └── ROCm vs Vulkan — depends on device and model
  │
  └── Which OS should you use?
      ├── Linux → Prefer headless environment
      │   └── Experimental ROCm drivers may conflict with GUI
      └── Windows → Prefer WSL2 for ROCm
          ├── WSL2 has production-level ROCm support
          └── Native Windows ROCm support is less mature than WSL2
```

## 1. Hardware Selection

### NPU Generations

| Generation | Architecture | Recommendation |
|------------|-------------|----------------|
| **X2** (Ryzen AI 300 series) | XDNA 2 | ✅ Prioritize NPU or NPU+GPU hybrid inference — typically fastest |
| **X1** (Ryzen 7000/8000 series) | XDNA 1 | ⚠️ Prioritize GPU instead — NPU adaptation is incomplete |
| **No NPU** | — | ✅ Use GPU (iGPU or dGPU with ROCm / Vulkan) |

**Rule of thumb:**
- X2 NPU → NPU or NPU+GPU hybrid (best performance)
- X1 NPU or no NPU → GPU (iGPU on this platform)

## 2. Framework Selection

### Step 1: Check Model Support on AMD Hugging Face

Visit [AMD's Hugging Face organization](https://huggingface.co/amd) to verify if your target model is officially supported.

| Status | Recommended Path |
|--------|-----------------|
| ✅ Model supported | **Ryzen AI toolchain** — the simplest way to leverage AMD hardware |
| ❌ Not supported | See Step 2 |

### Step 2a: With X2 NPU — Ryzen AI OGA Framework

If your device has an **X2 NPU** and the model is not on AMD's HF, use the **Ryzen AI OGA** (Open Generative AI) framework.

```
Model → [Olive / Quark Quantization] → OGA → NPU / NPU+GPU hybrid
```

- **Olive** (Microsoft): model optimization & quantization toolchain
- **Quark** (AMD): quantization toolkit for AMD hardware
- Output: model ready for NPU-only or NPU+GPU hybrid inference

### Step 2b: Without X2 NPU — GPU Framework

When using GPU, there are two primary paths:

#### ROCm Path

```
Model → ROCm (HIP) → OS → iGPU/dGPU
```

- **Official devices**: Check [ROCm supported GPUs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
- **Unofficial devices**: Community support via [TheRock](https://github.com/ROCm/TheRock) — now approaching stability
- Recommended for: most models and devices

#### Vulkan Path

```
Model → llama.cpp / vLLM / ... → Vulkan → OS → GPU
```

- Alternative to ROCm for GPU inference
- May be better suited depending on device and model
- Evaluate both if one path has issues

> **Choosing between ROCm and Vulkan**: Test both on your specific device + model combination. The optimal choice varies.

## 3. OS Selection

| OS | Recommendation | Reason |
|----|---------------|--------|
| **Linux** | Headless environment (no GUI) | Experimental ROCm drivers may conflict with desktop environments (e.g. Cinnamon failed to boot, error -22) |
| **Windows** | WSL2 for ROCm | WSL2 provides production-level ROCm support; native Windows ROCm is functional but not as mature |
| **Windows Native** | Fallback option | Works for PyTorch with proper setup but expect more issues than WSL2 |

### ROCm Support by OS

| Feature | Linux | Windows Native | WSL2 |
|---------|-------|---------------|------|
| ROCm GPU compute | ✅ Full | ⚠️ Partial (TheRock) | ✅ Production |
| GUI compatibility | ⚠️ May conflict | ✅ Native GUI | ✅ Via WSLg |
| NPU (Vitis AI) | ❌ Not available | ✅ | ❌ Not available |
| Recommended for | Headless GPU inference | NPU inference | ROCm GPU inference |

## Compatibility Matrix

### SmolVLM

| OS | Device | Framework | Attn Implementation | Status | Notes |
|----|--------|-----------|-------------------|--------|-------|
| Ubuntu | CPU | PyTorch | eager | ✅ Working | bf16 supported |
| Ubuntu | CPU | PyTorch | memory-efficient | ✅ Working | |
| Ubuntu | iGPU (ROCm) | PyTorch | eager | ✅ Working | bf16, ~8.8 GiB VRAM |
| Ubuntu | iGPU (ROCm) | PyTorch | memory-efficient | ✅ Working | |
| Ubuntu | iGPU (ROCm) | PyTorch | flash_attention_2 | ❌ Broken | Awaits new ROCm release |
| Ubuntu | iGPU (ROCm) | ONNX Runtime | — | ❌ Untested | MIGraphX EP facing issues |
| Ubuntu | NPU | — | — | ❌ N/A | No NPU support on Linux |
| Windows | CPU | PyTorch | eager | ✅ Working | bf16 |
| Windows | CPU | PyTorch | memory-efficient | ✅ Working | |
| Windows | iGPU (ROCm) | PyTorch | eager | ✅ Working | Requires HIP_VISIBLE_DEVICES=0 |
| Windows | iGPU (ROCm) | PyTorch | memory-efficient | ✅ Working | |
| Windows | iGPU (ROCm) | PyTorch | flash_attention_2 | ❌ Broken | Awaits new ROCm release |
| Windows | iGPU (ROCm) | ONNX Runtime | — | ❌ Broken | ROCm support incomplete on Windows |
| Windows | NPU | ONNX Runtime | — | 🔄 WIP | Vitis AI EP, being tested |
| Windows (WSL2) | CPU | PyTorch | eager | 🔄 Testing | |
| Windows (WSL2) | iGPU (ROCm) | PyTorch | eager | 🔄 Testing | |

### VGGT

| OS | Device | Framework | Status | Notes |
|----|--------|-----------|--------|-------|
| Ubuntu | CPU | PyTorch | ✅ Working | bf16 |
| Ubuntu | iGPU (ROCm) | PyTorch | ✅ Working | bf16, ~8.8 GiB VRAM |
| Ubuntu | iGPU (ROCm) | ONNX Runtime | ❌ Untested | Same MIGraphX issues |
| Ubuntu | NPU | — | ❌ N/A | |
| Windows | CPU | PyTorch | ✅ Working | |
| Windows | iGPU (ROCm) | PyTorch | ✅ Working | Requires HIP_VISIBLE_DEVICES=0 |
| Windows | iGPU (ROCm) | ONNX Runtime | ❌ Broken | ROCm support incomplete |
| Windows | NPU | ONNX Runtime | ⚠️ Not recommended | Model not from official ONNX repo |
| Windows (WSL2) | CPU | PyTorch | 🔄 Testing | |
| Windows (WSL2) | iGPU (ROCm) | PyTorch | 🔄 Testing | |

## Recommended Default Config

| Setting | Value |
|---------|-------|
| Framework | PyTorch |
| Data type | bf16 |
| Attention | `eager` (safest) or `memory-efficient` |
| Device | iGPU with ROCm (fallback to CPU if ROCm unavailable) |
| ROCm version | 7.2 (nightly wheels from `gfx110X-all` channel) |
| VRAM config | 16 GB in BIOS (UMA Frame Buffer Size) |