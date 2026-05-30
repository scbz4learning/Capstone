# Quantization on AMD Ryzen AI / APU

## Status Summary

On this platform (ROCm 7.13, Radeon 780M gfx1103), quantization support is **limited**. Only **AMD Quark** (PyTorch Eager Mode PTQ) works for inference, but it is **not practically beneficial**.

### Compatibility Matrix

| Library | Method | Status | Notes |
|---------|--------|--------|-------|
| **AMD Quark** (PyTorch) | INT8/INT4 weight-only PTQ | ✅ Works but not useful | On-the-fly dequant; no GPU kernels — slower, no memory savings, accuracy loss |
| **gptqmodel** (via optimum) | GPTQ INT4 | ⚠️ Partial | 2/24 layers quantized; rocBLAS fp16 GEMM crash on gfx1103 |
| **bitsandbytes** | 4/8-bit quantization | ❌ Fails | ROCm 7.2 .so ABI incompatible with 7.13 runtime; source compilation blocked by missing cmake targets |
| **torchao** | Any | ❌ Fails | Cutlass kernels are CUDA-only; PyTorch fallback paths missing key ops |
| **quanto** | Any | ❌ Untested | v0.2.0 latest; known API bugs in early versions |
| **ONNX Runtime** | q4f16 / int8 / bnb4 | ❌ Not recommended | Vision encoder bottleneck; MIGraphX version mismatch |

---

## Overall Conclusion: Weight-Only Quantization Is Not Beneficial Here

All quantization methods tested in this project are **weight-only** (W8A16 or W4A16): weights are stored in INT8/INT4 with scale/zero-point, while activations remain FP16/BF16. During each forward pass, weights are dequantized back to FP16 on-the-fly before the GEMM operation.

| Dimension | Claimed Benefit | Actual Result | Why |
|-----------|----------------|---------------|-----|
| **VRAM** | 2×–4× reduction | No reduction (4.18 GB stays 4.18 GB) | Quark retains original BF16 weights + adds scale tensors; weights are only logically "marked" for quantization |
| **Speed** | Faster due to smaller memory bandwidth | **Slower** (INT4 ~62s vs baseline ~30s for 32 tok) | Dequantize happens in PyTorch without GPU kernel support; extra CPU-side overhead |
| **Accuracy** | Minimal loss | Some loss, worse for INT4 on 2B models | Small models are more sensitive to quantization noise |

**When weight-only quantization matters:**
- When backed by GPU kernels (bitsandbytes' INT4 GEMM, Triton, or CUDA cores) that operate directly on quantized data — this requires either CUDA or a working ROCm kernel compilation pipeline, neither of which is available here
- When the model exceeds GPU memory (e.g., 13B+ models) — SmolVLM's 4.18 GB fits comfortably in 16 GB
- When using a framework with native quantized inference support (vLLM, llama.cpp, ExLlama)

**Bottom line:** On this platform, weight-only quantization via PyTorch provides no benefit and should not be used for SmolVLM. The model fits in GPU memory, runs fastest in native BF16, and any accuracy regression is unnecessary.

---

## 1. AMD Quark

### What We Knew Before Starting

The SmolVLM model card mentioned quantization support via bitsandbytes, torchao, and Quanto. The AMD ROCm docs mentioned AMD Quark as the recommended solution. We had Quark installed in `.venv` (Python 3.12) and wanted to test it with SmolVLM.

### Step-by-Step Attempt

#### Step 1: Install and Import

```python
# uv pip install amd-quark
from quark.torch.quantization.config.config import QConfig, QLayerConfig
from quark.torch.quantization import Int8PerTensorSpec
from quark.torch import ModelQuantizer
```

Result: Import succeeds, but triggers kernel compilation:

```
[QUARK-ERROR]: C++ kernel compile error
  fatal error: hipsparse/hipsparse.h: No such file or directory
  clang++: error: cannot find ROCm device library
```

**Diagnosis:** The `import quark.torch.kernel` triggers on-the-fly compilation of HIP kernels via `torch.utils.cpp_extension.load()`. This fails because:
- The TheRock SDK does not include `hipsparse` headers (only `hipruntime` headers)
- clang++ cannot find the ROCm device library bitcode without `--rocm-path` flag

**Resolution:** Not required. The kernel compilation error is caught and logged; Quark falls back to pure PyTorch ops. Basic PTQ works without them.

#### Step 2: Try the Simplest Quantization (INT8 weight-only)

```python
w8_spec = Int8PerTensorSpec(observer_method="min_max", symmetric=True,
    scale_type="float", round_method="half_even", is_dynamic=False).to_quantization_spec()
quant_config = QConfig(global_quant_config=QLayerConfig(weight=w8_spec))
quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model)
```

Result: **Crash during inference** — Conv2D `padding` parameter error:

```
TypeError: conv2d() received an invalid combination of arguments
  - tuple of (str, str, str, str, str) for padding parameter
```

**Diagnosis:** Quark's quantized Conv2D module (`quark.torch.quantization.nn.modules.quantize_conv`) wraps `F.conv2d()` but passes the `padding` argument incorrectly when the original module uses string-based padding like `padding="valid"`. SmolVLM's SigLIP vision encoder uses `padding="valid"`.

**Fix:** Exclude vision encoder and connector from quantization:

```python
quant_config = QConfig(
    global_quant_config=QLayerConfig(weight=w8_spec),
    exclude=["lm_head", "model.vision_model.*", "model.connector.*"],
)
```

We also attempted to use `LLMTemplate` registration with `exclude_layers_name` but found that building the `QConfig` directly with `exclude` was simpler and more reliable than the template system.

#### Step 3: Test INT4 Quantization

```python
from quark.torch.quantization import Int4PerGroupSpec

w4_spec = Int4PerGroupSpec(
    ch_axis=0, group_size=32,
    symmetric=True, scale_type="float",
    round_method="half_even", is_dynamic=False,
).to_quantization_spec()
```

Initial attempt failed with `TypeError: Int4PerGroupSpec.__init__() got an unexpected keyword argument 'observer_method'`. Unlike `Int8PerTensorSpec`, `Int4PerGroupSpec` does not accept `observer_method`. After removing it and adding the `ch_axis=0` parameter (required positional arg), it worked.

#### Step 4: Verify Deployed Model

We checked that quantized layers still reported `dtype=bfloat16` — confirming that weights are stored in their original dtype with separate quantization metadata, not packed in-memory.

### Key Code (Final Working Version)

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from quark.torch.quantization.config.config import QConfig, QLayerConfig
from quark.torch.quantization import Int4PerGroupSpec
from quark.torch import ModelQuantizer

model = AutoModelForImageTextToText.from_pretrained(
    "/home/bokai/capstone/models/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

w4_spec = Int4PerGroupSpec(
    ch_axis=0, group_size=32,
    symmetric=True, scale_type="float",
    round_method="half_even", is_dynamic=False,
).to_quantization_spec()

quant_config = QConfig(
    global_quant_config=QLayerConfig(weight=w4_spec),
    exclude=["lm_head", "model.vision_model.*", "model.connector.*"],
)

quant_model = ModelQuantizer(quant_config).quantize_model(model)
```

### Failure Modes Summary

| Failure | Symptom | Root Cause | Resolution |
|---------|---------|------------|------------|
| Kernel compilation | `hipsparse/hipsparse.h: No such file` | TheRock SDK missing devel headers | Ignore (non-fatal for PTQ) |
| Conv2D padding | `TypeError: conv2d() invalid padding` | Quantized Conv2D can't handle `padding="valid"` | Exclude vision encoder |
| INT4 spec | `TypeError: unexpected keyword argument` | `Int4PerGroupSpec` API differs from `Int8PerTensorSpec` | Fix parameters: no observer_method, add ch_axis |

---

## 2. GPTQ (gptqmodel + optimum)

### What We Knew Before Starting

We had already tested AMD Quark but wanted to try actual weight compression (not just on-the-fly). The ROCm docs mentioned GPTQ via AutoGPTQ. The Hugging Face transformers `GPTQConfig` provides a simple API. We created a new Python 3.10 environment (`.venv3`) for this since auto-gptq lacked CPython 3.12 wheels.

### Step-by-Step Attempt

#### Step 1: Install Dependencies

```bash
uv pip install auto-gptq --no-build-isolation \
  --extra-index-url https://huggingface.github.io/autogptq-index/whl/rocm573/
```

Result: Failed with `No matching distribution for Python 3.12`. We then switched to Python 3.10.

#### Step 2: Try GPTQConfig via transformers

```python
from transformers import GPTQConfig
quant_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
```

Result: `ImportError: cannot import name 'no_init_weights' from 'transformers.modeling_utils'`.

**Diagnosis:** `auto_gptq.modeling._base` imports `no_init_weights` from transformers, but this function was removed/renamed in transformers 5.x.

**Resolution:** Install `optimum` and `gptqmodel` instead. The new stack uses `optimum.gptq.GPTQQuantizer` which wraps `gptqmodel` rather than the old `auto-gptq`.

```bash
uv pip install optimum gptqmodel
```

#### Step 3: First Quantization Attempt (seqlen=2048)

```python
quantizer = GPTQQuantizer(
    bits=4, dataset="c4", group_size=128,
    model_seqlen=2048, batch_size=1,
    block_name_to_quantize="model.text_model.layers",
    modules_in_block_to_quantize=[
        ["self_attn.q_proj"], ["self_attn.k_proj"], ["self_attn.v_proj"],
        ["self_attn.o_proj"], ["mlp.gate_proj"], ["mlp.up_proj"], ["mlp.down_proj"],
    ],
)
```

Result: `ValueError: empty range for randrange() (0, 0, 0)`.

**Diagnosis:** The optimum data loader samples random sequences of length `model_seqlen` from the c4 dataset. The error means `input_ids.shape[1] - seqlen - 1 <= 0` — the tokenized text is shorter than the model sequence length. The model's tokenizer returned a different sequence length than expected. We didn't set `model_seqlen` explicitly in the first attempt.

**Fix:** Set `model_seqlen=1024`.

#### Step 4: Second Attempt (seqlen=1024)

Result: `HIPBLAS_STATUS_INTERNAL_ERROR` on block 1, layer `self_attn.o_proj`.

```
RuntimeError: CUDA error: HIPBLAS_STATUS_INTERNAL_ERROR
  when calling `hipblasGemmEx(..., HIP_R_16BF, ...)`
```

**Diagnosis:** The GPTQ calibration runs a forward pass through each transformer block to collect activation statistics. During this forward pass, a rocBLAS fp16 GEMM (bfloat16) fails on gfx1103. This is the same class of bug as the MIOpen FP16 Winograd error documented in known-issues.md.

**Fix attempt:** Switch to FP16 instead of BF16:

```python
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto",
)
```

#### Step 5: Third Attempt (FP16, seqlen=1024)

Result: **First two layers completed successfully!** Each layer took ~42 seconds for 7 submodules.

```
Layer 1/24: q_proj(4s) k_proj(4s) v_proj(4s) o_proj(4s) gate_proj(4s) up_proj(4s) down_proj(8s)
Layer 2/24: same pattern
```

Then on layer 3, `HIPBLAS_STATUS_INTERNAL_ERROR` again, this time with `rocblas_datatype_f16_r`. The rocBLAS fp16 GEMM crashed, this time during `mlp.down_proj` or `self_attn.o_proj` — the failure point was non-deterministic.

#### Step 6: Memory Optimization (seqlen=512)

Result: OOM at layer 3:

```
HIP out of memory. Tried to allocate 10.00 MiB.
GPU 0 has a total capacity of 16.00 GiB of which 0 bytes is free.
Of the allocated memory 15.06 GiB is allocated by PyTorch.
```

**Diagnosis:** GPTQ calibration requires storing activations for all calibration samples at each block boundary. With seqlen=512, batch_size=1, ~128 samples: the layer input cache is ~128 × 512 × 2048 × 2 bytes = 256 MB per hook, times several hooks. With the model (4.5 GB), Hessian matrices (16 MB × 7 modules × 24 layers = 2.7 GB), and PyTorch overhead, total exceeds 16 GB.

**Fix:** Reduce seqlen to 256:

```python
model_seqlen=256,
max_input_length=256,
cache_block_outputs=False,
```

Also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`.

#### Step 7: Fourth Attempt (FP16, seqlen=256)

Result: Same `HIPBLAS_STATUS_INTERNAL_ERROR` at layer 3, `self_attn.o_proj`. The memory settings helped avoid OOM, but the rocBLAS crash persisted.

### Failure Modes Summary

| Failure | Symptom | Root Cause | Status |
|---------|---------|------------|--------|
| Python version | No wheel for cp312 | auto-gptq lacks Python 3.12 ROCm wheels | Switched to Python 3.10 |
| transformers 5.x compat | `cannot import no_init_weights` | auto-gptq incompatible with transformers 5.x | Switched to gptqmodel |
| Empty calibration range | `randrange(0, 0, 0)` | model_seqlen not set or too large | Set model_seqlen=256 |
| rocBLAS GEMM crash | `HIPBLAS_STATUS_INTERNAL_ERROR` | gfx1103 fp16 GEMM bug in ROCm 7.13 | **Unresolved — hardware/ROCm limitation** |
| OOM | `HIP out of memory` | seqlen too large for 16 GB | Fixed with seqlen=256 |

### Key Code (Final Attempt)

```python
from optimum.gptq import GPTQQuantizer

quantizer = GPTQQuantizer(
    bits=4, dataset="c4", group_size=128,
    damp_percent=0.1, desc_act=False, sym=True,
    model_seqlen=256, batch_size=1,
    block_name_to_quantize="model.text_model.layers",
    modules_in_block_to_quantize=[
        ["self_attn.q_proj"], ["self_attn.k_proj"], ["self_attn.v_proj"],
        ["self_attn.o_proj"], ["mlp.gate_proj"], ["mlp.up_proj"], ["mlp.down_proj"],
    ],
    cache_block_outputs=False, max_input_length=256, pad_token_id=0,
)
quant_model = quantizer.quantize_model(model, tokenizer)
```

---

## 3. bitsandbytes

### What We Knew Before Starting

bitsandbytes is the most popular quantization library and is referenced by the SmolVLM model card. The ROCm docs mention a ROCm-aware fork at `https://github.com/ROCm/bitsandbytes`. The docs also note that pre-compiled binaries exist for ROCm 6.2–7.2, but newer versions require source compilation.

Our ROCm version is 7.13 (from TheRock), so we anticipated potential compatibility issues.

### Step-by-Step Attempt

#### Attempt 1: Direct pip install

```bash
uv pip install bitsandbytes
# → bitsandbytes==0.49.2
```

```python
import bitsandbytes as bnb
```

Result:
```
RuntimeError: Configured ROCm binary not found at
  .../bitsandbytes/libbitsandbytes_rocm83.so
```

**Diagnosis:** bitsandbytes encodes the ROCm version as `major × 10 + minor`. ROCm 7.13 → `7 × 10 + 13 = 83`. The pre-compiled binaries shipped with the wheel only go up to ROCm 7.2 (`rocm72`). Available: `rocm62, rocm63, rocm64, rocm70, rocm71, rocm72`.

List the available `.so` files:
```bash
ls .venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_rocm*
# → libbitsandbytes_rocm62.so ... libbitsandbytes_rocm72.so
```

**Hypothesis:** If the ROCm 7.2 binary has the same HIP kernel interface as 7.13, we could symlink it.

#### Attempt 2: Symlink + BNB_ROCM_VERSION

Create a symlink for version 0.50.0.dev0 format (`rocm713`) or override with env var:

```bash
# For bitsandbytes 0.49.2 (rocm encoding: 7*10+13=83):
ln -sf libbitsandbytes_rocm72.so libbitsandbytes_rocm83.so
export BNB_ROCM_VERSION=72
```

The newer dev version (0.50.0.dev0) supports `BNB_ROCM_VERSION` env var:

```bash
# Install from GitHub source (Python-only, no .so rebuild)
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes && pip install .
export BNB_ROCM_VERSION=72
```

```python
import bitsandbytes as bnb  # loads successfully with BNB_ROCM_VERSION=72
x = torch.randn(16, 16, device='cuda')
y = bnb.matmul(x, x.T, threshold=6.0)
```

Result:
```
MatMul8bitLt: inputs will be cast from torch.float32 to float16
Error invalid device function at line 532 in file /src/csrc/ops.hip
```

**Diagnosis:** `invalid device function` means the GPU kernel binary inside `libbitsandbytes_rocm72.so` contains device functions (HIP intrinsics) that are not available or have changed ABI in ROCm 7.13. The `.so` was compiled for ROCm 7.2's device library and runtime; loading it with ROCm 7.13 produces incompatible kernel code objects.

**Conclusion:** ABI compatibility between ROCm versions is not guaranteed. The HIP runtime API may be stable, but kernel device functions can change.

#### Attempt 3: Source Compilation (bitsandbytes-foundation)

```bash
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1103" -S . -B build
```

Result:
```
CMake Error: The HIP compiler identification is unknown
  does not contain the HIP runtime CMake package
  .../hip-lang/hip-lang-config.cmake
```

**Diagnosis:** The cmake `enable_language(HIP)` looks for `hip-lang-config.cmake`. The TheRock SDK installs ROCm as Python packages (e.g., `_rocm_sdk_core/`) and does not provide cmake module files for HIP.

**Fix attempt:** Create a stub `hip-lang-config.cmake` and set `CMAKE_HIP_COMPILER`:

```cmake
# Stub cmake config
set(CMAKE_HIP_COMPILER "${ROCM_PATH}/lib/llvm/bin/clang++")
set(CMAKE_HIP_COMPILER_ROCM_ROOT "${ROCM_PATH}")
```

But then cmake detected the wrapper:
```
CMAKE_HIP_COMPILER is set to the hipcc wrapper. This is not supported.
Use Clang directly, or let CMake pick a default.
```

Created a clang wrapper with `--rocm-path` and patched `CMakeTestHIPCompiler.cmake` to skip the test:

```bash
cat > /tmp/hip-clang-wrapper.sh << 'SCRIPT'
#!/bin/bash
ROCM_PATH=/home/bokai/.venv3/lib/python3.10/.../_rocm_sdk_core
exec $ROCM_PATH/lib/llvm/bin/clang++ \
  "--rocm-path=$ROCM_PATH" \
  "--rocm-device-lib-path=$ROCM_PATH/lib/llvm/amdgcn/bitcode" "$@"
SCRIPT

# Patch cmake HIP test to always succeed
cat > /tmp/my-cmake-modules/CMakeTestHIPCompiler.cmake << 'CMEOF'
set(CMAKE_HIP_COMPILER_WORKS 1 CACHE INTERNAL "")
set(CMAKE_HIP_COMPILER_FORCED 1 CACHE INTERNAL "")
CMEOF

cmake -DCOMPUTE_BACKEND=hip \
  -DCMAKE_HIP_COMPILER="/tmp/hip-clang-wrapper.sh" \
  -DCMAKE_MODULE_PATH="/tmp/my-cmake-modules" \
  -DCMAKE_HIP_ARCHITECTURES="gfx1103" ...
```

Result: Further errors:
```
Target "bitsandbytes" links to target "roc::hipblas" but the target was not found.
```

**Diagnosis:** The bitsandbytes CMakeLists.txt links against `roc::hipblas` (and other ROCm libraries). These require proper cmake imported targets defined by `hipblas-config.cmake` or find-modules, which TheRock does not provide. Creating stubs for `hipblas`, `rocblas`, `rocsparse`, `hiprand`, etc. with all required target definitions is impractical — each would need correct library paths, include dirs, and interface link dependencies matching the actual ROCm library ABI.

#### Attempt 4: Source Compilation (ROCm/bitsandbytes fork)

```bash
git clone --recurse https://github.com/ROCm/bitsandbytes.git
git checkout rocm_enabled
cd bitsandbytes
python setup.py install
```

Result: Installed Python files but did not compile any `.so` — the setup.py calls cmake internally which failed silently or was skipped. No `libbitsandbytes_rocm*.so` was produced.

**Diagnosis:** The ROCm fork's setup.py invokes cmake during `build_ext`, but without proper ROCm cmake infrastructure, the native extension build step silently produces nothing while the pure-Python portion installs successfully.

### Failure Modes Summary

| Attempt | What We Tried | Failure | Root Cause |
|---------|--------------|---------|------------|
| PyPI install | `pip install bitsandbytes` | No matching `.so` for ROCm 7.13 | Wheel only ships up to ROCm 7.2 |
| Symlink + env var | `BNB_ROCM_VERSION=72`, symlink `rocm72` as `rocm83` | `invalid device function` at runtime | Kernel ABI incompatible between ROCm 7.2 → 7.13 |
| Source compile (upstream) | `cmake -DCOMPUTE_BACKEND=hip` | cmake can't find HIP compiler | TheRock lacks cmake module files for HIP |
| Source compile (upstream, patched) | Stub cmake modules, clang wrapper | Missing `roc::hipblas` cmake target | TheRock lacks cmake target definitions for ROCm libs |
| Source compile (ROCm fork) | `python setup.py install` | No `.so` produced | setup.py cmake build step silently fails |

---

## 4. Quark Fast Kernel Compilation (Detailed)

This is not a separate quantization approach but a recurring issue: whenever `import quark.torch.kernel` is triggered (which happens during `from quark.torch import ModelQuantizer`), it tries to compile CUDA/HIP kernels on-the-fly.

### What It Tries to Do

```python
# Inside quark/torch/kernel/hw_emulation/extensions.py
from torch.utils.cpp_extension import load
load(
    name="kernel_ext",
    sources=[...hip files...],
    extra_cflags=["--offload-arch=gfx1103"],
)
```

This compiles HIP source files from `quark/torch/kernel/hw_emulation/csrc/` including:
- `mx/funcs.hip` — MX (Microscaling) dequantization
- `mxfp4/fake.hip`, `mxfp4/dequantize.hip` — MXFP4 quantization
- `tqt/tqt.hip` — Trained Quantization Threshold
- `fake_tensor_hip_hip.hip` — Fake quantization for tensor
- `python_function_export_hip.cpp` — Python bindings

### Failure Details

**Error 1 — hipsparse header missing:**
```
fatal error: hipsparse/hipsparse.h: No such file or directory
```
The TheRock SDK only ships `hip` headers (HIP runtime), not `hipsparse`, `hipblas`, `rocblas` etc. These are optional dependencies but `python_function_export_hip.cpp` includes `ATen/hip/HIPContext.h` which in turn includes `<hipsparse/hipsparse.h>`.

**Error 2 — ROCm device library not found:**
```
clang++: error: cannot find ROCm device library
  provide its path via '--rocm-path' or '--rocm-device-lib-path'
```
The TheRock clang++ doesn't know where the ROCm installation is — it's not at `/opt/rocm` and there's no default search path. The device library bitcode files exist at `$ROCM_PATH/lib/llvm/amdgcn/bitcode/` but clang++ doesn't look there without `--rocm-path`.

### Why This Is Non-Fatal

Quark catches the `load()` exception in `extensions.py` and logs it but continues. The kernel module is only needed for:
- MX format quantization (MXFP4, MXFP6)
- TQT (Trained Quantization Threshold)
- Fast fake-quantize operations

For basic PTQ (INT8, INT4 weight-only), Quark falls back to PyTorch-native quantize/dequantize operations, which are slower but functionally equivalent.

### What We Tried to Fix It

```bash
# 1. Set ROCM_PATH for hipcc to find device libraries
export ROCM_PATH="$VIRTUAL_ENV/.../_rocm_sdk_core"

# 2. Create clang wrapper with --rocm-path
/tmp/hip-clang-wrapper.sh ... --rocm-path=$ROCM_PATH ...

# 3. Fixed hip-lang cmake config
```

None of these worked completely because the compilation also needs `hipsparse/hipsparse.h` which is simply not distributed by TheRock's Python SDK. Installing the full ROCm devel stack (`rocm-dev`) would provide it, but that conflicts with TheRock's pip-installed ROCm and may break the GPU driver on this system.

---

## 5. Other Libraries

### torchao

**Status:** ❌ Not compatible

torchao's quantization kernels are implemented using NVIDIA's Cutlass library and compiled specifically for CUDA architectures. The PyTorch fallback paths (used when no CUDA/Cutlass is detected) only implement a subset of quantization schemes — notably missing INT4 groupwise quantization. There is no HIP/ROCm code path.

No workaround was attempted — the library explicitly targets CUDA and would require a full port to HIP.

### quanto

**Status:** ❌ Untested

The latest PyPI version is 0.2.0. The project's own documentation and issues tracker describe bugs where weights are reset to `None` during quantization in early 0.2.x releases. No newer version has been published. If the bugs were fixed, it would still face the same HIP kernel compilation issues as Quark and bitsandbytes.

### ONNX Runtime

ONNX Runtime offers pre-quantized model variants downloaded via `scripts/onnx/smolvlm-onnx.sh`:

| Variant | Size | Description |
|---------|------|-------------|
| `q4f16` | ~1.5 GB | 4-bit weights, FP16 activations |
| `fp16` | ~4 GB | Half precision |
| `int8` | ~3 GB | 8-bit integer |
| `bnb4` | ~1.6 GB | bitsandbytes 4-bit |

These were not retested during this quantization investigation session. Their limitations are documented in [ONNX Runtime](../Inference_Frameworks/onnxruntime.md): the vision encoder runs on CPU, negating any decoder quantization speedup.

---

## 6. Environment Setup

### ROCm Library Symlinks

The TheRock SDK installs ROCm shared libraries with versioned suffixes. Fix before running any quantization code:

```bash
source /home/bokai/capstone/scripts/rocm/fix_rocm_so.sh
```

This script (reproduced below) creates unversioned symlinks and sets `LD_LIBRARY_PATH`:

```bash
# fix_rocm_so.sh logic:
# For each lib*.so.* in _rocm_sdk_core/lib, create lib*.so -> lib*.so.* symlink
# Create version aliases: libamdhip64.so.6 -> libamdhip64.so.7
# Add all ROCm lib dirs to LD_LIBRARY_PATH
```

### Key Environment Variables

```bash
# REQUIRED: Path to TheRock ROCm SDK
export ROCM_PATH="$VIRTUAL_ENV/lib/python$(python3 -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/_rocm_sdk_core"

# For bitsandbytes: override ROCm version selection (dev branch only)
export BNB_ROCM_VERSION=72

# Debug: serialize GPU kernel execution for better error messages
export AMD_SERIALIZE_KERNEL=3
```

### Python Environments Tested

| Name | Python | ROCm | torch | Tested Approaches |
|------|--------|------|-------|-------------------|
| `.venv` | 3.12.3 | 7.13.26162 | 2.10.0+rocm7.13 | Quark |
| `.venv3` | 3.10.20 | 7.13.26183 | 2.10.0+rocm7.13 | Quark, GPTQ, bitsandbytes |

Python version does not affect quantization outcomes. Python 3.10 was chosen for GPTQ because auto-gptq lacked ROCm wheels for 3.12; in the end we used gptqmodel which supports both.

---

## 7. References

- [AMD Quark Documentation](https://quark.docs.amd.com/latest/)
- [gptqmodel GitHub](https://github.com/ModelCloud/GPTQModel)
- [bitsandbytes GitHub](https://github.com/bitsandbytes-foundation/bitsandbytes)
- [ROCm bitsandbytes Fork](https://github.com/ROCm/bitsandbytes)
- [SmolVLM-Instruct on Hugging Face](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- [ROCm Model Quantization Guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-quantization.html)
- [Known Issues on This Platform](../known-issues.md)
- [PyTorch on ROCm](../Inference_Frameworks/pytorch.md)
- [ONNX Runtime on ROCm](../Inference_Frameworks/onnxruntime.md)