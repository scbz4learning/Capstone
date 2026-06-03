# llama.cpp GPU Deployment Experience

## GPU Backend Support Matrix

| Capability | Vulkan | ROCm (HIP) |
|---|---|---|
| **gfx1103 Detection** | ✅ | ✅ (HIP runtime) |
| **Model Load** | ✅ | ✅ (tensors load to ROCm0) |
| **Inference** | ✅ | ❌ (kernel code object mismatch) |
| **Setup Complexity** | Simple (no extra deps) | Complex (LD_LIBRARY_PATH, HSA_OVERRIDE) |

## Current Status

### Vulkan (✅ Working)

Vulkan is the recommended backend for llama.cpp on gfx1103. It works out of the box with the prebuilt ROCm binary package (`llama-b9357`).

**Example:**
```bash
/home/bokai/capstone/third-party/llama-cpp/out/vulkan/llama-b9357/llama-cli \
  -m model.gguf --mmproj mmproj.gguf \
  -ngl 99 -p "hello" -n 10
```

### ROCm HIP (❌ Blocked)

The prebuilt ROCm binary (`llama-b9357`, compiled for ROCm 7.2) detects the GPU and loads tensors to ROCm0, but **inference crashes** with:

```
ROCm error: device kernel image is invalid
```

**Root Cause:** The `libggml-hip.so` code objects were compiled for `gfx1100`, `gfx1101`, `gfx1102` only. gfx1103 (Radeon 780M) is **not included**. When `HSA_OVERRIDE_GFX_VERSION=11.0.3` forces the runtime to load mismatched kernels, ISA validation fails.

This is purely a llama.cpp build-time issue — the prebuilt archives from the llama.cpp release page don't include gfx1103 in their `GPU_TARGETS`. The TheRock ROCm SDK itself supports gfx1103 (it's listed in the `gfx110X-all` per-family package).

## Environment Setup (for future use when upstream fixes this)

### Required Variables

```bash
# Source the fix script to add ROCm libs to LD_LIBRARY_PATH
source /home/bokai/capstone/scripts/rocm/fix_rocm_so.sh

# Force gfx1103 detection (kernel JIT only works if code objects include gfx1103)
export HSA_OVERRIDE_GFX_VERSION=11.0.3
export HIP_VISIBLE_DEVICES=0
```

### Source Build (for future use)

When upstream llama.cpp ships gfx1103 support, or to build your own:

```bash
# Requires TheRock tarball for cmake configs
# (PyPI packages don't include hip-config.cmake)
cmake -B build_hip -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1103 \
  -DCMAKE_C_COMPILER=hipcc \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_PREFIX_PATH=/path/to/therock-articraft
cmake --build build_hip -j$(nproc)
```

See [TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) for per-family package install instructions.

## Profiling Notes

- When profiling llama.cpp with Vulkan on gfx1103, use `measurement_mode: "gpu_plus_cpu_separate"` with `is_integrated: false` to capture separate GPU power via sysfs.
- The `RAPL_only_integrated_gpu` mode only measures CPU package power and should not be used for GPU backend benchmarks.
- Pre-built ROCm binary location: `third-party/llama-cpp/out/rocm/llama-b9357/`
- Pre-built Vulkan binary location: `third-party/llama-cpp/out/vulkan/llama-b9357/`
