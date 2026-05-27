# Drivers

## Ubuntu / Linux

### Install AMDGPU Driver

```bash
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt install -y ./amdgpu-install_7.2.70200-1_all.deb
sudo apt update
sudo apt install -y python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME
newgrp render video
sudo apt install -y rocm
```

### Check AMDGPU Driver

```bash
REQUIRED_AMDGPU="1:7.2.70200-2278374.24.04"
INSTALLED_AMDGPU=$(dpkg-query -W -f='${Version}' amdgpu 2>/dev/null || true)

if [ "$INSTALLED_AMDGPU" = "$REQUIRED_AMDGPU" ] && lsmod | grep -q amdgpu; then
  echo "✔ AMDGPU driver version $INSTALLED_AMDGPU is installed and module is loaded."
else
  echo "AMDGPU driver check failed. To reinstall:"
  echo "  sudo apt autoremove -y amdgpu-dkms"
  echo "  sudo apt install -y ./amdgpu-install_7.2.70200-1_all.deb"
  echo "  sudo apt update"
  echo "  sudo apt install -y \"linux-headers-$(uname -r)\" \"linux-modules-extra-$(uname -r)\""
  echo "  sudo apt install -y amdgpu-dkms"
  echo "  sudo update-initramfs -u -k all"
  echo "  sudo reboot"
fi
```

## Windows

Install the latest [Adrenalin driver](https://www.amd.com/en/products/software/adrenalin.html).

### WSL2

Install **AMD Software: Adrenalin Edition 26.2.2 for WSL2** or later to ensure proper GPU passthrough to the Linux subsystem.
