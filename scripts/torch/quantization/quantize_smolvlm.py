import torch
import time
import os
os.environ["QUARK_LOG_LEVEL"] = "warning"

MODEL_PATH = "/home/bokai/capstone/models/SmolVLM-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForImageTextToText, AutoProcessor

# ============================================================
# Step 1: Load model
# ============================================================
print(f"[1/5] Loading SmolVLM from {MODEL_PATH} ...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print(f"  Model: {model.config.architectures[0]}")
param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
print(f"  Memory (BF16): {param_size:.2f} GB")

# ============================================================
# Step 2: Configure Quark quantization
# ============================================================
from quark.torch.quantization.config.config import QConfig, QLayerConfig
from quark.torch.quantization import Int4PerGroupSpec

print(f"[2/5] Configuring INT4 weight-only quantization (group_size=32) ...")

w4_spec = Int4PerGroupSpec(
    ch_axis=0,
    group_size=32,
    symmetric=True,
    scale_type="float",
    round_method="half_even",
    is_dynamic=False,
).to_quantization_spec()

quant_config = QConfig(
    global_quant_config=QLayerConfig(weight=w4_spec),
    exclude=[
        "lm_head",
        "model.vision_model.*",
        "model.connector.*",
    ],
)

# ============================================================
# Step 3: Quantize
# ============================================================
from quark.torch import ModelQuantizer

print(f"[3/5] Quantizing ...")
quantizer = ModelQuantizer(quant_config)
t0 = time.time()
quant_model = quantizer.quantize_model(model)
t1 = time.time()
print(f"  Done in {t1-t0:.1f}s")

del model
torch.cuda.empty_cache()

# ============================================================
# Step 4: Verify quantization
# ============================================================
print(f"[4/5] Verification ...")
for name, module in quant_model.named_modules():
    if hasattr(module, "weight") and module.weight is not None:
        if name and "text_model.layers" in name:
            parts = name.split(".")
            if parts[-1] in ("q_proj", "k_proj", "v_proj", "o_proj") and parts[-3] in ("0", "23"):
                w = module.weight
                print(f"  {name}: dtype={str(w.dtype).split('.')[-1]}, shape={list(w.shape)}")
            if parts[-3] == "0" and parts[-1] == "q_proj":
                break  # only first match

# ============================================================
# Step 5: Test inference
# ============================================================
print(f"[5/5] Testing inference ...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
from torchvision.transforms import ToPILImage

image = ToPILImage()(torch.ones(3, 384, 384))
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image briefly."}
    ]},
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = {k: v.to(DEVICE, dtype=torch.bfloat16 if v.dtype.is_floating_point else v.dtype) for k, v in inputs.items()}

with torch.no_grad():
    t0 = time.time()
    out = quant_model.generate(**inputs, max_new_tokens=32, min_new_tokens=32)
    t1 = time.time()

text = processor.decode(out[0], skip_special_tokens=True)
print(f"  Output: {text[:100]}...")
print(f"  Latency: {(t1-t0)*1000:.0f}ms for 32 tokens")
print(f"\n=== Success! ===")