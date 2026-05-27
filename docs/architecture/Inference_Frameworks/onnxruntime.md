# ONNX Runtime on AMD Ryzen AI / APU

## Overview

ONNX Runtime offers potential optimizations (graph optimization, quantization) for LLM inference. However, on AMD APU systems, multiple technical barriers prevent it from being the recommended solution for multi-modal models.

## Why ONNX Runtime Is Not Recommended for VLM on AMD APU

### 1. Vision Encoder Becomes the Bottleneck

**The Problem:**

- ONNX Runtime can optimize the decoder (LLM) component, showing measurable speedup in token generation
- However, the vision encoder (for multimodal models) rarely has optimized backends available on AMD platforms
- This results in the vision encoder running on CPU via OpenVINO or similar, which is significantly slower than GPU acceleration

**Result:** The decoder optimization gains are negated by vision encoder slowdown, making the overall pipeline slower than native PyTorch inference.

### 2. No Full-GPU ONNX Pipeline

For ONNX Runtime to provide overall speedup on VLMs, all components (vision encoder + decoder) must run on GPU or NPU. Current limitations:

| Backend | Availability | Issue |
|---------|--------------|-------|
| `MIGraphXExecutionProvider` | AMD ROCm systems | Version incompatibility (see below) |
| `ROCMExecutionProvider` | Declared but unstable | Fallback to CPU on failure |
| `OpenVINOExecutionProvider` | Available | CPU-only optimization |
| `VitisAIExecutionProvider` | Ryzen AI NPU | No VLM support (see section 3) |

### 3. MIGraphX Version Compatibility Issues

**The Root Cause:**
MIGraphX (AMD's graph inference engine) suffers from version mismatches between:

- AMD's apt repositories providing pre-built MIGraphX binaries
- Your system's installed ROCm runtime version
- The ONNX Runtime Python bindings that depend on MIGraphX

**Specific Problems:**

| Scenario | Issue | Impact |
|----------|-------|--------|
| apt MIGraphX version ≠ ROCm pip version | HIP API incompatibility | JIT compilation errors or assertion crashes |
| Mix of old/new packages | Header version mismatch | Undefined behavior, graph compilation failures |
| Debug vs Release builds | Assertion crashes in debug builds | Runtime abort on first API call |

**Workaround:** Compile MIGraphX from source to match your exact ROCm version. This is complex and not recommended unless you have strong C++ build experience.

### 4. ONNX Model Export Quality

Export from Hugging Face may produce ONNX graphs with issues:

- **Graph optimization conflicts:** Models optimized for WebGPU / JavaScript runtime may conflict with ONNX Runtime graph optimization passes
- **Quantization format mismatch:** q4f16 or other quantization formats may not be supported by all execution providers
- **Missing operator support:** Operations like PerceiverResampler (used in vision connectors) may not be compilable to certain backends

**Recommendation:** Test ONNX model compatibility with your specific execution provider before investing time in deployment.

---

## NPU Inference via ONNX Runtime GenAI (OGA)

### Why NPU Cannot Run VLMs

Ryzen AI NPU accesses ONNX Runtime via `onnxruntime-vitisai` (OGA + VitisAI ExecutionProvider). Critical limitation:

**Model Support:**

- AMD only provides OGA-optimized models for **text-only LLMs** (SmolLM, Llama, Qwen)
- No pre-optimized OGA models exist for **multi-modal VLMs** (SmolVLM, LLaVA, etc.)

**Operator Coverage:**

- Complex vision connectors (e.g., PerceiverResampler for cross-attention) are not in the VitisAI-supported operator list
- Even if individual ops like Conv2d are supported, a full vision encoder + connector as a single ONNX model cannot be compiled

**Why Not Custom Export?**

- Converting a custom ONNX VLM to OGA format requires AMD Quark (proprietary tool), not publicly available
- Manual decomposition (vision encoder on iGPU, decoder on NPU) fails because `onnxruntime-rocm` and `onnxruntime-vitisai` Python modules cannot coexist in the same process

---

## Environment Configuration Issues

### Multiple ONNX Runtime Package Conflicts

Different packages install conflicting versions of the `onnxruntime` Python module:

```
onnxruntime           (PyPI standard)     → Providers: CPU only
onnxruntime-openvino                      → Providers: OpenVINO, CPU
onnxruntime-vitisai   (Ryzen AI)          → Providers: VitisAI, CPU
onnxruntime-rocm      (PyPI)              → Declared: MIGraphX, ROCM, CPU
onnxruntime-migraphx  (AMD apt repos)     → Declared: MIGraphX, CPU
```

**Impact:**

- `pip install` order determines which package's `onnxruntime` is active
- `get_available_providers()` returns only the final installed package's providers
- Switching between providers requires uninstalling/reinstalling packages, which is error-prone

**Best Practice:** Use PyTorch with ROCm directly instead of trying to manage multiple ONNX Runtime variants.

---

## Recommendation

For VLM inference on AMD Ryzen AI systems, **use PyTorch with native ROCm backend** instead of ONNX Runtime:

- Stable, well-documented, no package conflicts
- Vision encoder and decoder both benefit from GPU acceleration
- No custom compilation or version matching required
- Slightly slower token generation than ONNX (due to less graph optimization), but significantly faster overall due to GPU vision encoding

ONNX Runtime is only worth exploring if:
1. You can compile MIGraphX from source to match your ROCm version
2. You have pre-exported, tested ONNX models known to work with your setup
3. The 10–20% decoder speedup is critical for your use case
