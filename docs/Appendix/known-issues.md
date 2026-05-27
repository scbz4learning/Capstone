# Known Issues

ROCm support for **gfx1103** (Radeon 780M) is still in early stage. Bugs — both reproducible and intermittent — are common. Always check [TheRock issues](https://github.com/ROCm/TheRock/issues) for the latest status. Below are issues encountered during this project.

## 1. MIOpen FP16 Winograd Error

**Issue**: [TheRock #3044](https://github.com/ROCm/TheRock/issues/3044) — `HSA_ERROR_INVALID_ISA` when running FP16 operations (e.g. ResNet50 benchmark) on gfx1103. The specific kernel `miopenSp3AsmConvFury_v2_4_1_gfx11...` triggers an invalid ISA error.

**Workaround**:
```bash
export MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3=0
```

## 3. Flash Attention Unavailable (CK Missing)

Composable Kernel (CK) support is not yet available for gfx1103 in the current ROCm release. As a result, **flash attention 2 is unusable** — it falls back to `eager` or `memory-efficient` attention. The hardware should theoretically support it, but a new ROCm release is required.

## 4. Windows ROCm Not Detected (CI)

**Issue**: [TheRock #3905](https://github.com/ROCm/TheRock/issues/3905) — On Windows gfx110x, smoke tests fail with `Failed to retrieve GPU info: ERROR:ROCm is not available`. This is a CI/CD infrastructure issue affecting Azure runners.

## 5. HIP_VISIBLE_DEVICES Workaround (Resolved)

On Windows, iGPU enumeration could fail with `HIP API error = 0100 "no ROCm-capable device is detected"`. This required setting `HIP_VISIBLE_DEVICES=0` to force the HIP runtime to recognize the iGPU. This appears to be **resolved** in recent ROCm releases — no workaround is needed.