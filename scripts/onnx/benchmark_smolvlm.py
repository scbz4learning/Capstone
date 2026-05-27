#!/usr/bin/env python3
"""Quick ONNX benchmark: 1 warmup + 1 test, 2 images, 128 tokens."""

import json, os, sys, tempfile, time
import numpy as np
from PIL import Image
from transformers import AutoProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDIR = os.path.join(ROOT, "models", "SmolVLM-Instruct-onnx")
ODIR = os.path.join(MDIR, "onnx")

import onnxruntime as ort
print(f"ORT: {ort.__version__}")

def pick(prefix):
    for v in ["q4f16","fp16","int8","bnb4",""]:
        p = os.path.join(ODIR, f"{prefix}_{v}.onnx") if v else os.path.join(ODIR, f"{prefix}.onnx")
        if os.path.exists(p): return p
    return None

with open(os.path.join(MDIR, "config.json")) as f:
    cfg = json.load(f)
tc = cfg["text_config"]
IMG_TOK = cfg["image_token_id"]
EOS = tc.get("eos_token_id", 0)
NL = tc["num_hidden_layers"]
NKH = tc.get("num_key_value_heads", tc["num_attention_heads"])
HD = tc.get("head_dim", 64)

dp = pick("decoder_model_merged")
ep = pick("embed_tokens")
vp = pick("vision_encoder")
copt_ov = ort.SessionOptions(); copt_ov.enable_mem_pattern = False
copt_ov.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
copt_cpu = ort.SessionOptions(); copt_cpu.enable_mem_pattern = False

# Load processor
proc = AutoProcessor.from_pretrained(MDIR)

# 2 test images
images = [Image.new("RGB", (384,384), color=128) for _ in range(2)]
msgs = [{"role":"user","content":[{"type":"image"}] * 2 + [{"type":"text","text":"Describe the images briefly."}]}]
pr = proc.apply_chat_template(msgs, add_generation_prompt=True)
inp = proc(text=pr, images=images, return_tensors="np")
pv = inp["pixel_values"].astype(np.float32)
pm = inp.get("pixel_attention_mask")
if pm is not None: pm = pm.astype(np.bool_)
iid = inp["input_ids"].astype(np.int64)
am = inp["attention_mask"].astype(np.int64)
bs = 1

has_ov = "OpenVINOExecutionProvider" in ort.get_available_providers()
print(f"Loading models (vision: {'OpenVINO' if has_ov else 'CPU'}, decoder: CPU)...")
t0 = time.perf_counter()
vs = ort.InferenceSession(vp, copt_ov, providers=["OpenVINOExecutionProvider"] if has_ov else ["CPUExecutionProvider"])
es = ort.InferenceSession(ep, copt_cpu, providers=["CPUExecutionProvider"])
ds = ort.InferenceSession(dp, copt_cpu, providers=["CPUExecutionProvider"])
print(f"  Load: {time.perf_counter()-t0:.2f}s")

# Vision
print("Vision encoder...")
t0 = time.perf_counter()
vi = {"pixel_values": pv}
if pm is not None: vi["pixel_attention_mask"] = pm
feat = vs.run(["image_features"], vi)[0]
vt = time.perf_counter() - t0
print(f"  {vt*1000:.0f}ms")

# Decode
def gen(iid, am, feat):
    kv = {f"past_key_values.{l}.{k}": np.zeros([bs, NKH, 0, HD], dtype=np.float16)
          for l in range(NL) for k in ("key","value")}
    out = []
    ttft = None
    dts = []
    for step in range(128):
        t0 = time.perf_counter()
        emb = es.run(None, {"input_ids": iid.astype(np.int64)})[0]
        if step == 0:
            ip = np.where(iid[0] == IMG_TOK)[0]
            for i in range(min(feat.shape[0], len(ip))):
                emb[0, ip[i]] = feat[i, 0]
            if feat.shape[0] > len(ip):
                ex = feat[len(ip):, 0]
                emb = np.concatenate([emb, ex.reshape(1,-1,2048)], 1)
                am = np.concatenate([am, np.ones((1,ex.shape[0]), dtype=np.int64)], 1)
        din = dict(inputs_embeds=emb, attention_mask=am, **kv)
        logits, *pr = ds.run(None, din)
        p = logits[0,-1,:]
        probs = np.exp(p - np.max(p)); probs /= np.sum(probs)
        nid = np.array([[np.random.choice(len(probs), p=probs)]])
        tok = nid[0,0]
        el = time.perf_counter() - t0
        if step == 0: ttft = el * 1000
        else: dts.append(el * 1000)
        out.append(tok)
        if tok == EOS: break
        iid = nid.astype(np.int64)
        am = np.concatenate([am, np.ones((1,1), dtype=np.int64)], 1)
        kv = {k: pr[i] for i,k in enumerate(kv)}
    return out, ttft, dts

print("Warmup (1 iter)...")
gen(iid.copy(), am.copy(), feat)

print("Test (1 iter)...")
_, ttft, dts = gen(iid.copy(), am.copy(), feat)
tpot = np.mean(dts) if dts else 0

print(f"\n{'='*40}")
print(f"  2 images, 128 tokens")
print(f"{'='*40}")
print(f"  Vision:     {vt*1000:.0f} ms")
print(f"  TTFT:       {ttft:.0f} ms")
print(f"  TPOT mean:  {tpot:.1f} ms")
print(f"  Total:      {(vt*1000 + ttft + tpot * 128)/1000:.1f} s")
print(f"{'='*40}")

res = {
    "config": "onnx_cpu_q4f16", "num_images": 2, "max_new_tokens": 128,
    "vision_ms": round(vt*1000,1),
    "ttft_ms": round(ttft,1),
    "tpot_ms_mean": round(tpot,1),
    "total_ms": round(vt*1000 + ttft + tpot * 128, 1),
}
with open(os.path.join(ROOT, "benchmark_onnx_result.json"), "w") as f:
    json.dump(res, f, indent=2)
print(f"\nSaved to benchmark_onnx_result.json")