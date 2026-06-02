# Software Stack Quick Selection

## Decision Flowchart

```
Does your device have an NPU?
├── Yes → What is your NPU device?
│   ├── X2 NPU (STX/KRK, Ryzen AI 300 / AI Max 300)
│   │   └── What is your model type?
│   │       ├── CNN INT8 / CNN BF16 / NLP BF16 / Stable Diffusion → [Ryzen AI SW](#1-ryzen-ai-sw)
│   │       └── LLM → Is it in the OGA pre-optimized list ([AMD HF models](https://huggingface.co/amd/models))?
│   │           ├── Yes → Use [Lemonade](https://github.com/lemonade-sdk/lemonade), [llama.cpp](https://github.com/ggml-org/llama.cpp), or OGA directly
│   │           └── No → Are all operators supported ([ops table](https://ryzenai.docs.amd.com/en/latest/ops_support.html))?
│   │               ├── Yes → Consider OGA or ONNXRuntime with a custom patch
│   │               └── No → Does ROCm have an official driver for your device?
│   │                   ├── Yes → OGA or ONNXRuntime for GPU+NPU hybrid inference
│   │                   └── No → ONNXRuntime CPU+NPU hybrid, or other APIs
│   │                      ([LLM overview](https://ryzenai.docs.amd.com/en/latest/llm/overview.html))
│   │
│   └── X1 NPU (PHX/HPT, Ryzen AI 7000/8000)
│       └── Is your model CNN INT8 on Windows?
│           ├── Yes → [Ryzen AI SW](#1-ryzen-ai-sw)
│           └── No → NPU inference not available; fall back to GPU or CPU
│
└── No → Does your device have a GPU?
    ├── Yes → Does an official ROCm driver exist for your GPU?
    │   [Radeon compatibility matrices](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/compatibility.html)
    │   [Ryzen compatibility matrices](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/compatibility.html)
    │   ├── Yes → Use official driver. Choose any inference framework that supports ROCm or Vulkan; results may vary by model
    │   └── No → Do you have access to WSL2 (Windows Subsystem for Linux)?
    │       ├── Yes → Use WSL2 with librocdxg (ROCm DXG bridge)
    │       │   Routes GPU compute through the production-grade Windows AMD
    │       │   driver. Often significantly faster than TheRock for Conv-heavy
    │       │   models (e.g. VGGT: ~19× faster).
    │       │   See: https://github.com/ROCm/librocdxg
    │       │
    │       └── No → Check if any inference framework supports Vulkan for the model
    │           ├── Yes → Use Vulkan to drive GPU inference
    │           └── No → Try [TheRock](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md) community build for GPU inference
    │               Usually fine, even if slower — but note: TheRock's MIOpen is
    │               in early preview; Conv-heavy models may be significantly slower.
    │               If missing or problematic → fall back to CPU
    │
    └── No → CPU inference — see [CPU Inference](../Architecture/3_cpu-inference.md)
        Recommended frameworks: ZenDNN, OpenVINO, llama.cpp
```
