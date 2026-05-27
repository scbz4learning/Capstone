# NPU Inference

Before considering GPU or CPU, check whether your target model and device are supported for NPU inference.

### 1. Check Model Support

Visit the [Ryzen AI release notes](https://ryzenai.docs.amd.com/en/latest/relnotes.html) to see which models are officially supported by the Ryzen AI software stack.

### 2. Check Device Support

Visit the [model compatibility table](https://ryzenai.docs.amd.com/en/latest/relnotes.html#model-compatibility-table) to verify your NPU hardware is compatible with your target model.

### 3. Use ryzen-ai

If both model and device are supported, use the [ryzen-ai](https://ryzenai.docs.amd.com/en/latest/) framework — the simplest and most optimised path to NPU inference on AMD hardware.

!!! warning "Linux Compatibility"
    The Ryzen AI environment is **incompatible** with the ROCm stack provided by [TheRock](https://github.com/ROCm/TheRock) on Linux. If your target device lacks official ROCm support and your workload requires a hybrid NPU + GPU architecture, use **Windows** instead.

!!! warning ""
    This guide is based on Ryzen AI **1.7.1**. Information may be outdated. Always refer to the [official Ryzen AI documentation](https://ryzenai.docs.amd.com/en/latest/) for the latest supported models, devices, and installation instructions.

!!! tip "Advanced Users"
    Beyond `ryzen-ai`, you can deploy via ONNX Runtime with Vitis AI Execution Provider, direct Vitis AI flow, custom quantisation with AMD Quark or Microsoft Olive, or NPU+GPU hybrid execution. Before choosing a custom path, review the [operator support table](https://ryzenai.docs.amd.com/en/latest/ops_support.html) to verify your model's operators are compatible.