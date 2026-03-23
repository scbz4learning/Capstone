# APU Profiler (Capstone)

基于 `SmolVLM` 与 `VGGT` 的统一推理与性能分析框架。

特点：
- 支持 PyTorch + ONNX Runtime
- 支持 CPU / iGPU（未来可扩展 NPU）
- 内置 Profiling（TTFT / ITL / breakdown）
- 模块化设计：backend + model adapter + profiler

## 目录

- `configs/`: 模型配置
- `scripts/`: 运行脚本
- `apu_profiler/`: 核心代码
- `outputs/`: 性能输出

## 快速运行

```bash
pip install -r requirements.txt
python scripts/run_benchmark.py --model smolvlm --backend torch --device cuda
```

## 文件结构参考
请参考 `scripts/run_benchmark.py` 里的参数说明。
