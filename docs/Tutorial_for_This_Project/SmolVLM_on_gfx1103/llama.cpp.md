# llama.cpp

llama.cpp runs SmolVLM / SmolVLM2 with Vulkan offload on AMD Radeon 780M. No compilation or dependencies required — just download and run.

---

## Install

### 1. Download prebuilt binary (Ubuntu, Vulkan)

```bash
wget https://github.com/ggml-org/llama.cpp/releases/download/b9357/llama-b9357-bin-ubuntu-vulkan-x64.tar.gz
tar -xzf llama-b9357-bin-ubuntu-vulkan-x64.tar.gz
cd llama-b9357-bin-ubuntu-vulkan-x64
```

`llama-cli` is the inference binary. All required libraries are bundled; no `apt` installs needed.

### 2. Verify Vulkan works

```bash
./llama-cli --list-devices
```

You should see a `Vulkan` device listed. The AMD Radeon 780M (gfx1103) is supported natively.

---

## Run SmolVLM2-2.2B-Instruct

Use the SmolVLM2-Instruct GGUF + its `mmproj` (multimodal projector). Both are converted from the original Hugging Face checkpoint.

```bash
./llama-cli \
  -m SmolVLM2-2.2B-Instruct-f16.gguf \
  --mmproj SmolVLM2-2.2B-Instruct-mmproj-f16.gguf \
  -ngl 99 \
  --image /path/to/image.jpg \
  -p "Describe the image." \
  -n 128
```

| Flag | Meaning |
|---|---|
| `-m` | Language model GGUF |
| `--mmproj` | Vision encoder / projector GGUF |
| `-ngl 99` | Offload all layers to Vulkan (GPU) |
| `--image` | Input image path |
| `-p` | Text prompt |
| `-n` | Max new tokens |

For **lower memory**, use `Q4_K_M` or `Q8_0` quantization instead of `f16`. For **best accuracy**, use `f16`.

---

## Multiple images

Images are passed in order with the prompt:

```bash
./llama-cli \
  -m model.gguf \
  --mmproj mmproj.gguf \
  -ngl 99 \
  --image img1.jpg \
  --image img2.jpg \
  -p "Describe the two images." \
  -n 128
```

---

## Performance

On AMD Ryzen 7 8845HS + Radeon 780M:

| Model | Quant | TTFT | TPOT | Peak VRAM |
|---|---|---|---|---|---|
| SmolVLM2-2.2B-Instruct | f16 | ~117ms | ~47ms | ~5.4 GB |
| SmolVLM-Instruct | f16 | ~116ms | ~47ms | ~6.9 GB |

GPU VRAM usage reflects the full model size (3.2–6.9 GB depending on quantisation and model).

!!! warning "SmolVLM-Instruct tokenizer patch"
    The original `SmolVLM-Instruct` tokenizer is missing the `<global-img>` marker token and requires a small patch in `mtmd.cpp` (filter out `LLAMA_TOKEN_NULL`). **Use SmolVLM2-2.2B-Instruct instead** — it works out of the box with official llama.cpp builds.

!!! warning "ROCm backend"
    Do **not** use the ROCm backend. On this APU, ROCm silently falls back to CPU execution — performance and power equal CPU (~48-52W), and it can be up to 2.2× slower than CPU for FP16 due to ROCm overhead. Worse, the system can become **unstable** — power occasionally spikes to ~105W (over 2× normal) when ROCm is active, likely due to driver issues with the integrated GPU on gfx1103.
