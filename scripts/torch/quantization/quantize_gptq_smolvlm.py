import torch
import time
import os
os.environ["AMD_SERIALIZE_KERNEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

MODEL_PATH = "/home/bokai/capstone/models/SmolVLM-Instruct"

from transformers import AutoModelForImageTextToText, AutoTokenizer
from optimum.gptq import GPTQQuantizer

print(f"[1/3] Loading SmolVLM in FP16 (low mem) ...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()
mem_before = model.get_memory_footprint() / 1e9
print(f"  Memory before: {mem_before:.2f} GB")

print(f"\n[2/3] Applying GPTQ INT4 quantization (triton backend, low mem) ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

torch.cuda.empty_cache()
print(f"  GPU free after model load: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

quantizer = GPTQQuantizer(
    bits=4,
    dataset="c4",
    group_size=128,
    damp_percent=0.1,
    desc_act=False,
    sym=True,
    model_seqlen=256,
    batch_size=1,
    block_name_to_quantize="model.text_model.layers",
    modules_in_block_to_quantize=[
        ["self_attn.q_proj"],
        ["self_attn.k_proj"],
        ["self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj"],
        ["mlp.up_proj"],
        ["mlp.down_proj"],
    ],
    cache_block_outputs=False,
    max_input_length=256,
    pad_token_id=0,
)

t0 = time.time()
quant_model = quantizer.quantize_model(model, tokenizer)
t1 = time.time()
print(f"  Quantization done in {t1-t0:.1f}s")
print(f"  Memory after: {quant_model.get_memory_footprint() / 1e9:.2f} GB")

print(f"\n[3/3] Testing inference ...")
from transformers import AutoProcessor
from torchvision.transforms import ToPILImage

processor = AutoProcessor.from_pretrained(MODEL_PATH)
image = ToPILImage()(torch.ones(3, 384, 384))
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image briefly."}
    ]},
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    t0 = time.time()
    out = quant_model.generate(**inputs, max_new_tokens=32, min_new_tokens=32)
    t1 = time.time()

text = processor.decode(out[0], skip_special_tokens=True)
print(f"  Output: {text[:100]}...")
print(f"  Latency: {(t1-t0)*1000:.0f}ms for 32 tokens")

print(f"\n=== GPTQ Quantization Complete! ===")