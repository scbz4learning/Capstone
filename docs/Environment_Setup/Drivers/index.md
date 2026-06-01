# Drivers

## Ubuntu / Linux

### Install AMDGPU Driver

The Linux kernel usually includes the `amdgpu` driver by default, so a separate driver installation is generally not required.

If PyTorch does not run correctly or GPU detection fails, some users have reported that uninstalling `amdgpu-dkms` can help. See the discussion in [ROCm/TheRock issue #3618](https://github.com/ROCm/TheRock/issues/3618#issuecomment-4281691473) and the official AMD quick-start guide at https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html for the recommended recovery steps.

## Windows

Install the latest [Adrenalin driver](https://www.amd.com/en/products/software/adrenalin.html).

### WSL2

Install **AMD Software: Adrenalin Edition 26.2.2 for WSL2** or later to ensure proper GPU passthrough to the Linux subsystem.
