# Ryzen AI

Ryzen AI uses the **DirectML** execution provider on Windows for ONNX Runtime inference.

## Windows Setup

Ryzen AI software stack with DirectML enables GPU inference on AMD hardware via ONNX Runtime.

!!! warning "ROCm and Ryzen AI cannot coexist"
    ROCm and Ryzen AI **cannot be installed in the same virtual environment**. They are separate stacks.

## NPU

!!! note "X1 NPU on this device"
    The XDNA 1 (X1) NPU on this experimental device is **difficult to support** with current tooling. For experimental NPU support, refer to the [Developer Guide](../../../appendix/developer-guide.md).
