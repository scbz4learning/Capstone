# AMD Software Stack Architecture

This page maps the AMD software stack relevant to this project.

## Layer Structure

```
   ┌──────────────────────────────────────────────────────────────┐
   │                          Model                              │
   │               (SmolVLM, VGGT, ...)                          │
   └──────────────────────────┬───────────────────────────────────┘
                              │
   ┌──────────────────────────┼───────────────────────────────────┐
   │                   Inference Framework                       │
   ├────────────┬─────────────┼──────────────┬──────────────────┤
   │  PyTorch   │ ONNX Runtime│ llama.cpp    │      IREE        │
   │            │             │  vLLM ...    │  (experimental)  │
   └──────┬─────┴──────┬──────┴──────┬───────┴────────┬─────────┘
          │            │             │                │
   ┌──────┴─────┐ ┌────┴──────┐ ┌───┴────┐     ┌─────┴──────────┐
   │   ROCm     │ │ Ryzen AI  │ │ Vulkan │     │  MLIR (multi-  │
   │   (HIP)    │ │ (Vitis AI)│ │ (API)  │     │   level IR)    │
   └──────┬─────┘ └────┬──────┘ └───┬────┘     └─────┬──────────┘
          │            │             │                │
   ┌──────┴────────────┴─────────────┴────────────────┴───────────┐
   │                        OS Layer                              │
   ├────────────────────────────┬─────────────────────────────────┤
   │         Linux              │            Windows              │
   └────────────────────────────┴─────────────────────────────────┘
          │            │             │                │
   ┌──────┴────────────┴─────────────┴────────────────┴───────────┐
   │                       Hardware                               │
   ├─────────────────┬───────────────────┬───────────────────────┤
   │   CPU (x86-64)  │  iGPU (RDNA 3)    │   NPU (XDNA AIE)     │
   │                 │  Radeon 780M      │   Ryzen AI NPU        │
   │                 │  gfx1103          │                       │
   └─────────────────┴───────────────────┴───────────────────────┘
```

## Path Breakdown

### 1. PyTorch

```
Model → PyTorch ─┬→ CPU ──────────→ Windows / Linux ──→ CPU
                  │
                  └→ GPU ──→ ROCm (HIP) ──→ Windows / Linux ──→ iGPU (gfx1103)
```

### 2. ONNX Runtime

```
Model → ONNX Runtime ─┬→ CPU EP ────────────────→ Windows / Linux ──→ CPU
                       │
                       ├→ DirectML EP ───────────→ Windows ──────────→ iGPU
                       │
                       ├→ MIGraphX EP ───────────→ Linux ────────────→ iGPU
                       │
                       └→ Vitis AI EP ──→ Ryzen AI ──→ Windows / Linux ──→ CPU / GPU / NPU
                                                        (Vitis AI: Windows only)
```

### 3. Vulkan (llama.cpp, vLLM, etc.)

```
Model → llama.cpp / vLLM / ... ──→ Vulkan ──→ Windows / Linux ──→ GPU
```

### 4. IREE (Experimental)

```
Model ──→ IREE ──→ MLIR ──┬→ CPU    (stable)
                           ├→ GPU    (stable)
                           └→ NPU    (experimental)
                                │
                                └→ Windows / Linux (both supported)
```

## Ryzen AI — Cross-Layer Driver

Ryzen AI serves as a unified driver layer that not only powers the **Vitis AI EP** (ONNX Runtime) but also supports other frameworks directly:

| Framework / API | Ryzen AI Support |
|----------------|------------------|
| ONNX Runtime (Vitis AI EP) | ✅ |
| Lemon API | ✅ |
| JAX | ✅ |
| TensorFlow | ✅ |
| Others | ✅ |

Ryzen AI supports all OS and device targets:

```
Ryzen AI ─┬→ Windows ─┬→ CPU
           │           ├→ GPU
           │           └→ NPU
           │
           └→ Linux   ─┬→ CPU
                       ├→ GPU
                       └→ NPU
```

## Full Linkage Summary

| Inference Framework | Path to Hardware | Driver Layer | OS | Target Device | Status |
|---|---|---|---|---|---|
| **PyTorch** | CPU → OS → CPU | — | Linux / Windows | CPU | ✅ |
| **PyTorch** | GPU → ROCm → OS → iGPU | ROCm (HIP) | Linux / Windows | iGPU | ✅ |
| **ONNX Runtime** | CPU EP → OS → CPU | — | Linux / Windows | CPU | ✅ |
| **ONNX Runtime** | DirectML EP → OS → iGPU | DirectML | Windows | iGPU | ✅ |
| **ONNX Runtime** | MIGraphX EP → OS → iGPU | MIGraphX | Linux | iGPU | ⚠️ Facing issues |
| **ONNX Runtime** | Vitis AI EP → Ryzen AI → OS → HW | Ryzen AI (Vitis AI) | Windows | CPU / GPU / NPU | 🔄 WIP |
| **llama.cpp / vLLM / ...** | Vulkan → OS → GPU | Vulkan | Linux / Windows | GPU | 🔄 Planned |
| **IREE** | IREE → MLIR → HW | MLIR | Linux / Windows | CPU / GPU (stable), NPU (experimental) | 🔄 Experimental |