# Known Issues

!!! info "More issues"
    More issues can be found in [Related_Links.md](Related_Links.md) under **Issues Related** section.

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

## 6. VGGT Conv Performance Degradation on Linux Native ROCm (gfx1103)

**Issue**: VGGT runs ~20× slower on native Linux ROCm (~30.3s/image) compared to WSL (~1.56s/image) with the same BF16-SDPA configuration.

**Root Cause**: TheRock's MIOpen lacks optimized Conv kernels for `gfx1103`. The missing `.fdb.txt` kernel database causes `Conv2d`/`ConvTranspose2d` to fall back to slow software paths.

**Scope**: Affects Conv-heavy models (VGGT, DPTHead). Transformer-heavy models (e.g. SmolVLM) are unaffected since their compute is dominated by GEMM/attention.

**Workaround**:
- Use WSL, where `librocdxg` provides access to the production Windows AMD driver with a complete kernel database
- Use CPU inference on Linux as a fallback (~35.8s/image)

**Reference**: See [VGGT Overview](../../Tutorial_for_This_Project/VGGT_on_gfx1103/VGGT_Overview.md) for deployment profiles.

