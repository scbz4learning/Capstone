# BIOS & OS Prerequisite

The following settings and system requirements are essential before installing any drivers or acceleration backends.

## BIOS Configuration

### VRAM (UMA Frame Buffer Size)

For optimal performance with Large Language Models, it is highly recommended to fix the VRAM (integrated GPU memory) to **16GiB**.

- **Configuration Path**: `Bios` -> `Advanced` -> `AMD CBS` -> `NBIO` -> `GFX config`
- **Recommended Setting**: 16GiB (if available)

!!! info "Why 16GiB UMA?"
    According to our profiling logs for representative models like **SmolVLM** and **VGGT**, the system requires substantial memory overhead beyond just the model weights:

    - **SmolVLM Optimization**: Our tests show a peak CUDA reserved memory of approximately **10.6GiB** when running in bfloat16 mode.
    - **VGGT Optimization**: For models like VGGT, peak memory reservation can exceed **5.5GiB** even for relatively small vision tasks.

## Operating System Requirements

### Linux

See [ROCm system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#rdna-os) for the most up-to-date support list.

**Tested & Supported:**

- Ubuntu 24.04 LTS Server (Headless/No GUI)

**Known Issues:**

- **Desktop Environments (GUIs)**: We **strongly discourage** using any Linux desktop environment. Testing has confirmed that installing the `amdgpu` driver results in critical conflicts and boot failures with **Cinnamon**, **MATE**, and **GNOME**. For maximum stability and performance, a **headless server** environment is highly recommended.

### Windows

- **Minimum OS**: Windows 11 (22H2 or later recommended)
- **WSL2**: Required for certain Linux-native backends. Ensure WSL2 is up to date via `wsl --update`.
