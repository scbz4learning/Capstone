# llama.cpp

llama.cpp runs SmolVLM / SmolVLM2 with Vulkan offload on AMD Radeon 780M. No compilation or dependencies required — just download and run.

---

## Install

### 1. Download prebuilt binary (Ubuntu, Vulkan)

```bash
wget https://github.com/ggml-org/llama.cpp/releases/download/b9468/llama-b9468-bin-ubuntu-vulkan-x64.tar.gz
tar -xzf llama-b9468-bin-ubuntu-vulkan-x64.tar.gz
cd llama-b9468-bin-ubuntu-vulkan-x64
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
|---|---|---|---|---|
| SmolVLM2-2.2B-Instruct | f16 | ~116ms | ~47ms | ~0.22 GB |
| SmolVLM-Instruct | f16 | ~116ms | ~47ms | ~0.20 GB |

Vulkan offloading drops host memory from several GB (CPU-only) to ~0.2 GB.

!!! warning "SmolVLM-Instruct tokenizer patch"
    The original `SmolVLM-Instruct` tokenizer is missing the `<global-img>` marker token and requires a small patch in `mtmd.cpp` (filter out `LLAMA_TOKEN_NULL`). **Use SmolVLM2-2.2B-Instruct instead** — it works out of the box with official llama.cpp builds.

!!! warning "ROCm backend"
    Do **not** use the ROCm backend (`-ngl` is Vulkan-only in this setup). ROCm on this APU performs on par with CPU while drawing far more power (~105 W vs ~48 W).
