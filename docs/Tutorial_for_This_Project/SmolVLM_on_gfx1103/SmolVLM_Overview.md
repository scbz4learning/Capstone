# SmolVLM Overview

SmolVLM is a compact vision-language model (2B params) developed by Hugging Face, designed for on-device multimodal inference. For this project, we use it as a perception model — describing scenes, answering questions about images, and processing visual input for robotics applications.

---

## Quick Start

```bash
source venv/bin/activate

# Describe an image
python scripts/smolvlm/run_smolvlm.py \
    --images path/to/image.jpg \
    --prompt "Describe the scene briefly." \
    --device cuda --dtype bfloat16 \
    --output-dir smolvlm_output

# Run on CPU
python scripts/smolvlm/run_smolvlm.py \
    --images path/to/image.jpg \
    --prompt "What hazards do you see?" \
    --device cpu --dtype float32 \
    --output-dir smolvlm_output_cpu

# Stream output token-by-token
python scripts/smolvlm/run_smolvlm.py \
    --images path/to/image.jpg \
    --prompt "Describe this driving scene." \
    --device cuda --dtype bfloat16 \
    --stream

# Outputs: console output + result.json
```

## Framework Recommendation

| Framework | Recommendation | Reason |
|-----------|---------------|--------|
| **llama.cpp** (Vulkan) | ✅ Primary | Fastest TTFT (~95ms) + lowest power (~26-33W) for SmolVLM2-2.2B-Instruct with Q8_0 or Q4_K_M. Memory 3.2-5.4GB depending on quantization. Vulkan offloading critical. |
| **llama.cpp** (ROCm) | ⚠️ Not recommended | Same power as CPU (~48-52W), but 2.2× slower for f16 and no better for quantized. Effectively CPU execution with ROCm overhead. |
| **PyTorch** (BF16, SDPA) | ⚠️ Fallback only | Only choice for full-precision training or gradient-based use. ~6930ms TTFT — ~60× slower than llama.cpp Vulkan. |
| **ONNX Runtime** | ❌ Avoid | Image encoding is CPU-only on this platform. MIGraphX support incomplete on Linux. |

!!! tip "Production Recommendation"
    Use **SmolVLM2-2.2B-Instruct** (proper llama.cpp support, no patch needed, ~93ms TTFT with Q8_0 Vulkan, 3.9GB memory) over SmolVLM-Instruct. The original's tokenizer lacks `<global-img>` and requires a quick-fix patch.

!!! tip "Quantization Priority"
    - **Q8_0** — best balance: fastest TTFT (~93ms), 3.9GB memory, good accuracy
    - **Q4_K_M** — lowest memory (3.2GB) and best efficiency (1.86 tok/J), but slight accuracy loss — use for memory-constrained or latency-critical workloads
    - **f16** — best accuracy, but highest TTFT (~117ms) and memory (5.4GB)

---

## Key Profiling Chart

The following relative chart compares all configurations for SmolVLM-Instruct (2.2B) on Linux. Lower is better.

![SmolVLM-Instruct TTFT ratio](../../assets/profiling/smolvlm_instruct_ttft_ratio.png)

For the full family (all sizes), the same pattern holds:

![SmolVLM Family TTFT ratio](../../assets/profiling/smolvlm_family_dtype_ttft_ratio.png)

---

## Architecture Relevance

SmolVLM's backbone is **SmolLM2 1.7B**, a pure text transformer. Images are **not processed inside the backbone**: the vision encoder (SigLIP) converts each 384×384 image patch into **81 visual tokens**, which are concatenated with text tokens and fed into the backbone.

This means:

- Frameworks that only support text-only SmolLM **can be adapted** for VLM with manual image preprocessing
- But **full end-to-end VLM support** (image encoding + generation) is currently only available in **PyTorch**

---

## Model Details

| Property | Value |
|----------|-------|
| Parameters | 2B |
| Backbone | SmolLM2 1.7B |
| Visual tokens per patch | 81 |
| Max context | 16k tokens |
| License | Apache 2.0 |
| Paper | [arxiv: 2504.05299](https://arxiv.org/abs/2504.05299) |

## Citation

```bibtex
@article{marafioti2025smolvlm,
  title={SmolVLM: Redefining small and efficient multimodal models},
  author={Andr{'e}s Marafioti and Orr Zohar and Miquel Farr{'e} and Merve Noyan and
          Elie Bakouch and Pedro Cuenca and Cyril Zakka and Loubna Ben Allal and
          Anton Lozhkov and Nouamane Tazi and Vaibhav Srivastav and Joshua Lochner and
          Hugo Larcher and Mathieu Morlon and Lewis Tunstall and Leandro von Werra and
          Thomas Wolf},
  journal={arXiv preprint arXiv:2504.05299},
  year={2025}
}
```