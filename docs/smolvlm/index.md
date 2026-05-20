# SmolVLM

SmolVLM is a compact vision-language model (2B params) developed by Hugging Face, designed for on-device multimodal inference. For this project, we use it as a perception model — describing scenes, answering questions about images, and processing visual input for robotics applications.

## Architecture Relevance

SmolVLM's backbone is **SmolLM2 1.7B**, a pure text transformer. Images are **not processed inside the backbone**: the vision encoder (SigLIP) converts each 384×384 image patch into **81 visual tokens**, which are concatenated with text tokens and fed into the backbone.

This means:
- Frameworks that only support text-only SmolLM **can be adapted** for VLM with manual image preprocessing
- But **full end-to-end VLM support** (image encoding + generation) is currently only available in **PyTorch**

## Framework Recommendation

| Framework | Recommendation | Reason |
|-----------|---------------|--------|
| **PyTorch** | ✅ Recommended | Complete GPU/iGPU pipeline via Hugging Face `transformers` |
| **ONNX Runtime** | ⚠️ Not recommended | Image encoding is CPU-only on this platform — very slow. On Linux, MIGraphX support is also incomplete, making ONNX Runtime nearly unusable for GPU inference. |

See the [PyTorch page](0.pytorch.md) for sample inference code and benchmark results.

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