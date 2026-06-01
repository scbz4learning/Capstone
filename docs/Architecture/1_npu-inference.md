# NPU Inference

Before considering CPU or GPU, check whether your target model and device are
supported for NPU inference through the decision flowchart below.

## Decision Flowchart

```
What is your NPU device?
├── X2 NPU (STX/KRK, Ryzen AI 300 / AI Max 300)
│   └── What is your model type?
│       ├── CNN INT8 / CNN BF16 / NLP BF16 / Stable Diffusion → [Ryzen AI SW](#1-ryzen-ai-sw)
│       └── LLM → Is it in the OGA pre-optimized list ([AMD HF models](https://huggingface.co/amd/models))?
│           ├── Yes → Use [Lemonade](https://github.com/lemonade-sdk/lemonade), [llama.cpp](https://github.com/ggml-org/llama.cpp), or OGA directly
│           └── No → Are all operators supported ([ops table](https://ryzenai.docs.amd.com/en/latest/ops_support.html))?
│               ├── Yes → Consider OGA or ONNXRuntime with a custom patch
│               └── No → Does ROCm have an official driver for your device?
│                   ├── Yes → OGA or ONNXRuntime for GPU+NPU hybrid inference
│                   └── No → ONNXRuntime CPU+NPU hybrid, or other APIs
│                      ([LLM overview](https://ryzenai.docs.amd.com/en/latest/llm/overview.html))
│
└── X1 NPU (PHX/HPT, Ryzen AI 7000/8000)
    └── Is your model CNN INT8 on Windows?
        ├── Yes → [Ryzen AI SW](#1-ryzen-ai-sw)
        └── No → NPU inference not available; fall back to GPU or CPU
```

---

## 1. Ryzen AI SW

The official AMD software stack for production NPU inference. Provides
end-to-end model compilation and execution via ONNX Runtime with the Vitis AI
Execution Provider (VAI EP).

**Repository:** [amd/RyzenAI-SW](https://github.com/amd/RyzenAI-SW)
**Documentation:** [ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/latest/)

### Hardware Support (Ryzen AI 1.7.1)

| Processor | NPU Generation | Linux | Windows | CNN INT8 | CNN BF16 | NLP BF16 | LLM (OGA) |
|-----------|---------------|-------|---------|----------|----------|----------|-----------|
| Ryzen AI 7000/8000 (PHX/HPT) | X1 | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Ryzen AI 300 / AI Max 300 (STX/KRK) | X2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Key constraints:**

- **X1 NPU (PHX/HPT):** Only CNN INT8 is supported (Windows only). BF16, NLP
  BF16, LLM, and Stable Diffusion are NOT supported. Linux is NOT supported.
- **X2 NPU (STX/KRK):** INT8, BF16, NLP BF16, and LLM via OGA are all
  supported on both Linux and Windows.
- On STX/KRK, the `xclbin` option is deprecated for INT8 models; use the
  default configuration files instead.
- On PHX/HPT INT8 models, set `target=X1` and specify `xclbin` explicitly.

### LLM Execution Modes (STX/KRK only)

| Mode | Framework | Compute Allocation |
|------|-----------|-------------------|
| NPU-Only | OGA | NPU exclusive |
| Hybrid | OGA | Dynamic NPU + iGPU |
| GPU | llama.cpp | Dedicated iGPU |

### Stable Diffusion (STX/KRK only)

- Text-to-Image, Image-to-Image, ControlNet, dynamic shapes
- SD-2.1-v, SD-3.x/3.5-Turbo, Segmind-Vega

### High-Level LLM APIs

- **[Lemonade](https://github.com/lemonade-sdk/lemonade)** — open-source local
  AI server with Python SDK, REST API, CLI, and desktop GUI. Supports XDNA2 NPU
  via `flm`/`ryzenai-llm` backends. Multi-vendor, portable across execution
  backends.
- **[ONNX Runtime GenAI (OGA)](https://github.com/microsoft/onnxruntime-genai)**
  — native C++ and Python APIs for hybrid and NPU-only LLM execution.
- **[llama.cpp C++ Headers](https://github.com/ggml-org/llama.cpp)**
  — lightweight C++ inference engine supporting CPU, GPU, and XDNA2 NPU via
  the Ryzen AI SW backend.

### Getting Started

1. Verify hardware support in the [model compatibility table](https://ryzenai.docs.amd.com/en/latest/relnotes.html#model-compatibility-table).
2. Install Ryzen AI Software 1.7.1+ ([Linux guide](https://ryzenai.docs.amd.com/en/latest/), [Windows guide](https://ryzenai.docs.amd.com/en/latest/)).
3. For pre-optimized LLMs, browse the [AMD HuggingFace collection](https://huggingface.co/amd/models) and use Lemonade or OGA.
4. For custom models, check the [operator support table](https://ryzenai.docs.amd.com/en/latest/ops_support.html) and use AMD Quark or Microsoft Olive for quantization.
5. For hybrid NPU+iGPU execution, see the [LLM overview](https://ryzenai.docs.amd.com/en/latest/llm/overview.html).

---

## 2. MLIR-AIE (IRON)

A low-level, close-to-metal NPU programming toolkit. Provides Python APIs to
explicitly control AI Engine cores, memory hierarchy, and DMA data movement.
**Not an end-to-end model compilation flow.**

**Repository:** [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie)
**Language:** C, MLIR, C++, Python
**License:** Apache-2.0

### Scope and Limitations

- Designed for **operator-level programming**, not full model inference.
- Requires manual tiling, dataflow orchestration, and kernel fusion by the
  programmer.
- The largest reference examples are ResNet-class models; no transformer or
  LLM end-to-end flows are provided.
- Does not replace Ryzen AI SW; it complements it by exposing NPU internals
  for research and custom kernel development.
- Linux only (Ubuntu 24.04+).
- Requires separate installation of the XDNA driver (`amdxdna-dkms`) and the
  Peano/LLVM-AIE compiler.

### When to Use

- Research: exploring NPU architecture, tiling strategies, and custom kernel
  implementations.
- Prototyping: validating individual operators or small kernel sequences on NPU.
- **Not suitable for:** deploying full models, especially transformer-based
  architectures, without significant manual engineering.

---

## 3. IREE

An MLIR-based multi-hardware compiler and runtime. The `iree-amd-aie` plugin
adds experimental AMD NPU targeting to IREE.

**Repository:** [iree-org/iree](https://github.com/iree-org/iree)
**Plugin:** [nod-ai/iree-amd-aie](https://github.com/nod-ai/iree-amd-aie)
**License:** Apache-2.0

### Status

**Not usable for transformer-based models.** As confirmed by hands-on testing,
the `iree-amd-aie` plugin lacks support for the attention operators required by
transformer architectures (e.g., `GroupQueryAttention`, `RotaryEmbedding`).
Models such as SmolVLM, SmolLM, and other LLMs cannot be compiled or executed
through this path.

Requires:
- `xdna-driver` (specific commit)
- `llvm-aie` (Peano) compiler
- Optional: Vitis AIE Essentials (Chess) for best performance

### When to Use

- Research into compiler-level NPU backend integration.
- Non-transformer workloads with fully supported operators.
- **Do not use for production or transformer model deployment**; there is no
  end-to-end path and critical operators are missing.

---

## Summary

| Path | Readiness | X1 (PHX/HPT) | X2 (STX/KRK) | Transformer | Platform |
|------|-----------|--------------|--------------|-------------|----------|
| Ryzen AI SW | Production | INT8 CNN only (Win) | Full support | STX/KRK only | Win (+STX/KRK Linux) |
| MLIR-AIE / IRON | Research | Not viable | Limited operator-level | ✗ | Linux only |
| IREE | Experimental | Not viable | Not viable | ✗ | Linux only |

If your target model and device are supported by Ryzen AI SW on X2, that is
the recommended and only practical path to NPU acceleration on AMD hardware.
The other two paths are suitable only for research into low-level NPU
programming and compiler integration, and neither currently supports
transformer-based workloads.
