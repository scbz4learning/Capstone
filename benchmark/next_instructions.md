好，这里我给你一份**完整、工程级、可直接落地的《amd-ai-prof 统一 Profiling 框架设计指南》**。
这不是建议合集，而是**一套你可以按步骤实现的蓝图**。

---

# 🧠 0. 项目最终目标（先定清楚）

> 🎯 **amd-ai-prof = 面向 AMD / ROCm 的统一 AI 推理 Profiling 框架**

支持：

* ✅ PyTorch（forward + generative）
* ✅ ONNX Runtime（forward）
* ✅ ONNX Runtime GenAI（generative）
* ✅ 统一指标（latency / TTFT / TPOT / energy）
* ✅ 精确功耗分析（ROCm）

---

# 🧱 1. 总体架构（核心设计）

```
amd-ai-prof
│
├── core/              # 通用逻辑（最稳定）
├── backends/          # runtime 插件（torch / onnx / genai）
├── runners/           # 推理范式（forward / generative）
├── monitor/           # 硬件监控（ROCm）
├── metrics/           # 指标计算
├── trace/             # profiler trace
├── factory.py         # 构建入口
```

---

# 🔌 2. Backend 设计（关键抽象）

## 🎯 目标

> 👉 **不同 runtime = 插件，不影响核心逻辑**

---

## 2.1 PyTorch Backend

```python
class TorchBackend:
    def load_model(self, model_fn, device, dtype):
        model = model_fn().to(device).eval()
        return model

    def prepare_inputs(self, input_fn, device):
        return input_fn().to(device)

    def is_generative(self, model):
        return hasattr(model, "generate")
```

---

## 2.2 ONNX Runtime Backend

```python
class ONNXBackend:
    def load_model(self, model_path):
        import onnxruntime as ort

        return ort.InferenceSession(
            model_path,
            providers=["ROCMExecutionProvider", "CPUExecutionProvider"]
        )

    def run(self, session, inputs):
        return session.run(None, inputs)
```

---

## 2.3 ONNX GenAI Backend（重点）

👉 使用 onnxruntime-genai

```python
class ONNXGenAIBackend:
    def load_model(self, model_path):
        import onnxruntime_genai as og

        self.model = og.Model(model_path)
        self.tokenizer = og.Tokenizer(self.model)

        return self.model
```

---

# 🧠 3. Runner 抽象（最关键）

👉 这是你整个系统的“灵魂”

---

## 3.1 BaseRunner

```python
class BaseRunner:
    def run_step(self): pass
    def finalize(self, results): pass
```

---

## 3.2 ForwardRunner（统一 torch + onnx）

```python
class ForwardRunner(BaseRunner):
    def __init__(self, backend, model, inputs, device, dtype):
        self.backend = backend
        self.model = model
        self.inputs = inputs
        self.device = device
        self.dtype = dtype

    def run_step(self):
        t0 = time.perf_counter()

        if isinstance(self.backend, TorchBackend):
            with torch.no_grad(), torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.dtype
            ):
                self.model(self.inputs)

        else:
            self.backend.run(self.model, self.inputs)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        return {"latency_ms": (time.perf_counter() - t0) * 1000}

    def finalize(self, results):
        lat = np.mean([r["latency_ms"] for r in results])
        return {
            "type": "forward",
            "latency_ms": lat,
            "throughput": 1000 / lat
        }
```

---

## 3.3 GenerativeRunner（torch + onnx-genai）

```python
class GenerativeRunner(BaseRunner):
    def __init__(self, backend, model, inputs, device, dtype, max_new_tokens):
        self.backend = backend
        self.model = model
        self.inputs = inputs
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
```

---

### 👉 PyTorch 路径

```python
def _run_torch(self):
    t0 = time.perf_counter()

    self.model.generate(**self.inputs, max_new_tokens=1, use_cache=True)
    torch.cuda.synchronize()

    ttft = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    self.model.generate(**self.inputs, max_new_tokens=self.max_new_tokens, use_cache=True)
    torch.cuda.synchronize()

    total = (time.perf_counter() - t1) * 1000

    tpot = (total - ttft) / (self.max_new_tokens - 1)

    return {"ttft_ms": ttft, "tpot_ms": tpot}
```

---

### 👉 ONNX GenAI 路径（更精确 ⭐）

```python
def _run_onnx_genai(self):
    gen = self.model.create_generator()
    gen.append_prompt(self.inputs["prompt"])

    token_times = []

    for _ in range(self.max_new_tokens):
        t0 = time.perf_counter()

        gen.generate_next_token()

        dt = (time.perf_counter() - t0) * 1000
        token_times.append(dt)

    return {
        "ttft_ms": token_times[0],
        "tpot_ms": np.mean(token_times[1:]),
        "token_times": token_times
    }
```

---

# ⚡ 4. ROCm Monitor（功耗核心）

```python
class ROCmMonitor:
    def __init__(self, interval=0.05):
        self.stats = []
        self.stop_event = threading.Event()

    def _monitor(self):
        cmd = ["rocm-smi", "--showpower", "--showuse", "--json"]

        while not self.stop_event.is_set():
            t = time.time()

            try:
                out = subprocess.run(cmd, capture_output=True, text=True)
                data = json.loads(out.stdout)
                gpu = next(iter(data.values()))

                self.stats.append({
                    "t": t,
                    "power": float(gpu.get("Current Socket Graphics Package Power (W)", 0)),
                    "util": float(gpu.get("GPU use (%)", 0))
                })
            except:
                pass

            time.sleep(0.05)
```

---

# 🔥 5. 能量计算（必须用积分）

```python
def integrate_energy(stats):
    energy = 0

    for i in range(1, len(stats)):
        dt = stats[i]["t"] - stats[i-1]["t"]
        p = (stats[i]["power"] + stats[i-1]["power"]) / 2
        energy += p * dt

    return energy
```

---

# 🧠 6. UnifiedProfiler（核心调度）

```python
class UnifiedProfiler:
    def __init__(self, runner, monitor):
        self.runner = runner
        self.monitor = monitor

    def profile(self, warmup=2, steps=5):
        results = []

        self.monitor.start()

        for i in range(warmup + steps):
            out = self.runner.run_step()

            if i >= warmup:
                results.append(out)

        stats = self.monitor.stop()

        summary = self.runner.finalize(results)

        energy = integrate_energy(stats)
        duration = stats[-1]["t"] - stats[0]["t"]

        summary.update({
            "energy_j": energy,
            "avg_power_w": energy / duration
        })

        return summary
```

---

# 🧪 7. Trace 支持（独立）

```python
def capture_trace(model, inputs, device, dtype, file):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ]
    ) as prof:
        model(inputs)

    prof.export_chrome_trace(file)
```

---

# 🏗️ 8. Factory（统一入口）

```python
def build_profiler(backend_type, model, inputs, device, dtype):

    if backend_type == "torch":
        backend = TorchBackend()
        model = backend.load_model(model, device, dtype)
        mode = "generative" if backend.is_generative(model) else "forward"

    elif backend_type == "onnx":
        backend = ONNXBackend()
        model = backend.load_model(model)
        mode = "forward"

    elif backend_type == "onnx-genai":
        backend = ONNXGenAIBackend()
        model = backend.load_model(model)
        mode = "generative"

    if mode == "forward":
        runner = ForwardRunner(backend, model, inputs, device, dtype)
    else:
        runner = GenerativeRunner(backend, model, inputs, device, dtype, 30)

    return UnifiedProfiler(runner, ROCmMonitor())
```

---

# 📊 9. 统一输出格式（必须统一）

```json
{
  "type": "generative",
  "latency_ms": null,
  "ttft_ms": 120,
  "tpot_ms": 15,
  "throughput": 65,
  "avg_power_w": 110,
  "energy_j": 3.2
}
```

---

# ⚡ 10. AMD 专属优化（关键）

## ✅ dtype

```python
if torch.version.hip:
    dtype = torch.bfloat16
```

---

## ✅ attention

```python
attn_impl = "eager"
```

---

## ✅ ONNX IO Binding（强烈建议）

否则 latency 不准：

```python
session.io_binding()
```

---

# 🚀 11. 你最终能得到什么能力

### ✅ 支持

* CV 模型（VGGT）
* VLM（SmolVLM）
* LLM（未来）
* Torch vs ONNX 对比

---

### ✅ 高级能力

* TTFT vs TPOT 分析
* token latency 曲线（ONNX GenAI）
* energy / token
* AMD GPU 利用率分析

---

# 🎯 最终总结（一句话）

> 👉 **amd-ai-prof = 一个统一 profiling 框架 + 三类 backend + 两种 runner**

---

# 🚀 如果你下一步要做（强烈建议）

可以继续升级：

* CLI（`amd-ai-prof run config.yaml`）
* 自动画图（论文级）
* Torch vs ONNX 对比报告
* token-level power 曲线

---

如果你愿意，我可以下一步直接帮你：

👉 写一个**可运行的最小版本 repo（含 torch + onnx-genai）**
👉 或直接帮你做 CLI 工具

只要说一句你要哪种 👍
