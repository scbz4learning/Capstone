# SmolVLM Overview

SmolVLM is a compact vision-language model (2B params) developed by Hugging Face, designed for on-device multimodal inference. For this project, we use it as a perception model — describing scenes, answering questions about images, and processing visual input for robotics applications.

---

## Framework Recommendation

| Framework | Recommendation | Reason |
|-----------|---------------|--------|
| **llama.cpp** (Vulkan) | ✅ Primary | Best TTFT (~116ms) + lowest memory (~0.2GB) for SmolVLM2-2.2B-Instruct with Q4_K_M. f16 is preferred for accuracy. Vulkan offloading critical. |
| **llama.cpp** (ROCm) | ⚠️ Not recommended | Same speed as CPU, draws ~2× more power (~105W). No advantage over Vulkan. |
| **PyTorch** (BF16, SDPA) | ⚠️ Fallback only | Only choice for full-precision or gradient-based use. ~6930ms TTFT — ~60× slower than llama.cpp. |
| **ONNX Runtime** | ❌ Avoid | Image encoding is CPU-only on this platform. MIGraphX support incomplete on Linux. |

!!! tip "Production Recommendation"
    Use **SmolVLM2-2.2B-Instruct** (proper llama.cpp support, no patch needed, ~95ms TTFT with Q4_K_M Vulkan) over SmolVLM-Instruct. The original's tokenizer lacks `<global-img>` and requires a quick-fix patch.

!!! tip "Quantization Priority"
    - **f16** — best accuracy, memory ~0.2GB (Vulkan), ~116ms TTFT
    - **Q8_0** — near-f16 latency, memory ~0.2GB (Vulkan), ~95ms TTFT
    - **Q4_K_M** — fastest, ~95ms TTFT, but accuracy loss — use only for latency-critical, accuracy-tolerant workloads

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