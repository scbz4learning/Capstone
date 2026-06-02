# CPU Inference

## Overview

CPU inference is the universal fallback when no GPU or NPU path is
available or practical. It works on almost any hardware, but is
significantly slower than accelerated paths.

## When to Use

- No GPU acceleration path is available
- Model operators or frameworks lack GPU backend support
- Running in a constrained environment where GPU drivers cannot be installed

## Recommended Frameworks for AMD CPU

### 1. ZenDNN

AMD's official deep neural network library optimized for Zen architecture.
Plugs into ONNX Runtime as an execution provider and also integrates with
TensorFlow and PyTorch.

**Repository:** [AMD ZenDNN](https://www.amd.com/en/developer/zendnn.html)
**Documentation:** [ZenDNN docs](https://www.amd.com/en/developer/zendnn/documentation.html)

#### Key Capabilities

- Optimized kernel library for AMD Zen CPUs
- ONNX Runtime Execution Provider integration
- Support for INT8 quantization and BF16 inference
- Graph-level optimizations for CNN and transformer models

#### Installation

```bash
# Install via pip (ZenDNN package includes ONNX Runtime with ZenDNN EP)
pip install zendnn
```

Or build from source following the [official guide](https://www.amd.com/en/developer/zendnn.html).

---

### 2. OpenVINO

Intel's open-source inference framework with good AMD CPU support.
Strong for vision, NLP, and transformer models. Supports model
optimization via quantization and graph transformations.

**Repository:** [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)
**License:** Apache-2.0

#### Key Capabilities

- Model optimizer for graph-level transformations
- INT8, FP16, and FP32 inference precision
- Good performance on AMD Ryzen CPUs
- Broad model format support (ONNX, TensorFlow, PyTorch via export)

#### Installation

```bash
pip install openvino openvino-dev
```

#### Quick Start

```python
import openvino as ov

core = ov.Core()
model = core.read_model("model.xml")
compiled = core.compile_model(model, "CPU")
```

---

### 3. llama.cpp

Lightweight C/C++ inference engine optimized for CPU execution.
Excellent for LLM inference on AMD CPUs with support for quantization,
KV-cache optimizations, and continuous batching.

**Repository:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
**License:** MIT

#### Key Capabilities

- Highly optimized CPU kernels with GGML backend
- Support for Q4_0, Q4_K_M, Q5_K_M, Q8_0, and other quantization formats
- K-quant and importance-aware quantization
- Continuous batching and speculative decoding
- Low memory footprint compared to full-precision frameworks

#### Installation

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

Or install via pip:

```bash
pip install llama-cpp-python
```

#### Usage

```bash
./build/bin/llama-cli -m model.gguf -p "Hello" -n 128
```

---

### 4. ONNX Runtime (CPU EP)

ONNX Runtime with the CPU execution provider provides graph-level
optimizations and cross-platform inference. Can be paired with ZenDNN
for additional AMD CPU kernel acceleration.

**Repository:** [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
**License:** MIT

#### Key Capabilities

- Graph optimizations (fusion, constant folding, layout optimization)
- INT8 and FP16 quantization support
- Cross-platform (Linux, Windows)
- ZenDNN Execution Provider for AMD-specific tuning

#### Installation

```bash
pip install onnxruntime
# With ZenDNN EP:
pip install zendnn
```

---

### 5. PyTorch + oneDNN

PyTorch on CPU uses oneDNN (formerly MKL-DNN) for kernel optimization.
`torch.compile` provides additional graph-level fusion. While oneDNN
is Intel-originated, it performs well on AMD Zen CPUs.

**Repository:** [pytorch/pytorch](https://github.com/pytorch/pytorch)

#### Key Capabilities

- oneDNN kernel library for CPU inference
- `torch.compile` with dynamic shape support
- INT8 quantization via `torch.ao.quantization`
- Flexible model authoring and experimentation

#### Usage

```python
import torch

model = torch.load("model.pth")
model.eval()

# Enable oneDNN (default on most builds)
torch.backends.mkldnn.enabled = True

# Compile for additional optimization
compiled = torch.compile(model)
```

---

## Summary

| Framework | Best For | Quantization | AMD CPU Optimized |
|-----------|----------|-------------|-------------------|
| ZenDNN | ONNX models, CNNs, BERT | INT8, BF16 | Yes |
| OpenVINO | Vision, NLP, transformers | INT8, FP16 | Good |
| llama.cpp | LLMs | Q4-Q8, K-quant | Yes |
| ONNX Runtime | Cross-platform, graph-optimized | INT8, FP16 | Via ZenDNN EP |
| PyTorch + oneDNN | Research, flexible workflows | INT8 | Adequate |

For most AMD CPU workloads:
- **LLMs** → llama.cpp
- **ONNX models / CNNs** → ZenDNN + ONNX Runtime
- **Vision / NLP pipelines** → OpenVINO