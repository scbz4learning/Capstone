# Changelog

## W3

## W4


### Ongoing 
```python
直接结论／建议
- 这些项目（PyTorch / vLLM / SGLang / ONNX Runtime）都支持多层次的 profiling：从系统级（nsys/Nsight）→ 进程/线程级（pprof、system traces）→ GPU kernel/驱动级（CUPTI/nsys）→ 框架/操作级（ORT 的 session profiler、torch.profiler）→ 层/模块级（NVTX 或 layerwise hooks）→ 内存 / FLOPS / shapes / 堆栈信息。要覆盖它们提供的接口，你需要同时准备「代码级标记 + 配置/触发接口 + 采集/导出/合并 + 可视化/解析工具」这些要素。下面给出分层说明、实际 checklist 与示例命令/代码，方便你把一个模型的 profiling 覆盖到所有接口。

我已查看并归纳了 vLLM / SGLang / ONNX Runtime / PyTorch repo 中的实现与文档（找到了配置项、wrapper、NVTX hook、start_profile HTTP 接口、SessionOptions.enable_profiling 等），接下来把这些归纳成可执行的检查表和示例。下一步我可以根据你用哪个框架/模型，给出一份可直接运行的启动脚本和收集命令。

1) 支持的 profiling 级别（从粗到细）
- 系统/主机级（System-wide）
  - 工具：Nsight Systems / nsys、perf/top、system profilers
  - 作用：捕获进程创建、上下文切换、CPU/GPU利用率、跨进程依赖、子进程追踪
- GPU kernel / 驱动级
  - 工具：nsight compute / ncu、CUPTI via nsys
  - 作用：单核 kernel 性能、寄存器/带宽/occupancy/roofline 数据
- 进程/堆栈 / CPU sampling
  - 工具：pprof（Go/Go 服务）、python cProfile、perf、采样工具
- 框架级 Timeline / Trace（chrome-trace / perfetto）
  - 工具：torch.profiler 导出 trace、ORT 内置 profiler 输出 chrome-trace JSON、Perfetto UI
  - 作用：操作/节点/阶段的时间线、跨设备事件（可以在 perfetto/ui.perfetto.dev 查看）
- 层/模块/操作级（可视化 trace 中的条目）
  - 方法：NVTX range / layerwise hooks、torch.profiler.record_function / record_shapes、ORT Profiler events、custom NVTX hooks
  - 作用：查看每一层/每个 op 的时间、参数、shape、堆栈
- 内存/资源视角
  - 内容：显存峰值 / 分配/释放（torch.profiler profile_memory、ORT 内部统计、系统内存采样）
- 矩阵乘/内核策略 profiling（GEMM/内核选择）
  - ORT 中有 gemm_profiler、vLLM/others 也有专门 profiling 选择策略
- 合并 / 跨节点
  - 要点：把各 worker/local_rank 的 trace 合并，或用集中式采集（vLLM/sGLang 提供 start_profile/manager 支持分布式采集）

2) 要覆盖上面框架接口，你需要准备的清单（Checklist）
- A. 代码级准备（必须）
  - 在模型/推理路径中添加或打开 framework 的记录点：
    - PyTorch: 在关键段落（forward/backward、critical modules）使用 torch.profiler.record_function 或 register hooks。
    - NVTX: 为层/模块添加 NVTX range（或使用框架提供的 layerwise NVTX hook）。
    - ORT: 如果需要更细粒度，在自定义 Provider / Op 周期处使用 ORT Profiler API 或 NVTX 宏。
  - 如果框架已有 hook（比如 SGLang 的 nvtx_pytorch_hooks、vLLM 的 layerwise profile），确保在启动时开启对应 flag（--enable-layerwise-nvtx-marker 等）。
- B. 配置与触发（必须）
  - 提供可以打开/关闭 profiling 的配置或 CLI（例如：--profile、--profiler-config JSON、SessionOptions.enable_profiling=True）。
  - 支持按阶段/步数控制（warmup、wait、active、num_steps、start_step），避免一次性生成超大 trace。
  - 提供手动触发接口（HTTP /start_profile、/stop_profile 或 local API），并能把 trace 写到可访问目录。
- C. 环境与依赖（必须）
  - GPU profiling 需要 CUPTI / kineto 支持，确保运行时包或系统已安装（官方 wheel 通常内置 kineto；nsys 需要单独安装）。
  - 安装 Nsight Systems / Nsight Compute 如果要做 kernel 级分析。
  - 对 Go/服务端：开启 pprof（HTTP /debug/pprof/*）以便 CPU/heap 分析。
- D. 输出格式与保存（必须）
  - 支持保存为：chrome-trace JSON（.trace.json 或 .json.gz）、nsys-rep、perfetto 格式、pprof protobuf、ORT 的 profile JSON 等。
  - 支持 gzip 压缩与文件命名（worker 名称 / local_rank / profile_prefix / timestamp）。
- E. 收集/合并/解析工具（强烈建议）
  - 提供自动化脚本：触发 profile、等待步数完成、下载/合并 traces（vLLM、SGLang 已有示例脚本）。
  - 提供解析工具：将 raw trace 转成 perfetto 可视化或生成 layerwise 表格（vLLM/tools、sglang scripts、ORT 的 parse 脚本）。
- F. 可视化与分析（必须）
  - Perfetto UI (https://ui.perfetto.dev/)、TensorBoard（torch.profiler -> tensorboard_trace_handler）、chrome://tracing、nsys stats / export。
- G. 分布式/多卡注意（如果适用）
  - 把每个 rank 的 trace 放到独立文件夹并记录 rank → 合并时保留 pid/tid/worker_name。
  - 增大 RPC timeout / flush timeout（vLLM 文档提到需要延长 VLLM_RPC_TIMEOUT），避免 trace flush 过程超时。
- H. 基准与上下文信息（非常重要）
  - 记录模型版本、参数、batch size、输入 shape、硬件（GPU 型号/驱动/CUDA 版本）、随机种子、是否启用 cuda-graph、是否使用混合精度等。
  - 记录未 profile 的 baseline 运行时间以便对比。

3) 实操指引 + 最小示例（按框架）
- PyTorch（单机/实验）
  - 推荐用 torch.profiler，示例参数：activities=[CPU, CUDA], schedule=wait/warmup/active, record_shapes, profile_memory, with_stack。
  - snippet:
    ```
    from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=torch.profiler.schedule(wait=1,warmup=1,active=3),
                 on_trace_ready=tensorboard_trace_handler("logs"),
                 record_shapes=True, profile_memory=True, with_stack=True) as prof:
        for step in range(N):
            with record_function("forward"):
                out = model(inp)
                loss = out.sum()
            prof.step()
    ```
  - 可视化：tensorboard --logdir=logs 或 将 trace 导入 ui.perfetto.dev。
- vLLM
  - 配置文件或 CLI: --profiler-config '{"profiler": "torch","torch_profiler_dir":"./vllm_profile", "torch_profiler_with_stack": true, ...}'
  - 启动 server 或 bench: `vllm serve <model> --profiler-config '…'` 或 `vllm bench serve --profile`
  - 注意：使用 schedule 参数（warmup/active/wait），并设置 VLLM_RPC_TIMEOUT 足够大以等待 flush。
  - 会生成 layerwise trace，可用 vllm 的 tools/profiler/print_layerwise_table.py 解析。
- SGLang
  - 本地生成：`sglang generate --model-path <model> --profile [--profile-all-stages]`（docs/diffusion/profiling.md）
  - 服务端 profiling：使用 nsys 启动 server 或使用 `python -m sglang.profiler` 脚本通过 /start_profile 触发采集。
  - 若需 layer-wise：加 `--enable-layerwise-nvtx-marker` 并用 nsys capture，或用内置 profiler manager（profile_by_stage）。
  - 可视化：ui.perfetto.dev、nsys export。
- ONNX Runtime (ORT)
  - Python session-level profiling:
    ```
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.enable_profiling = True
    sess = ort.InferenceSession(model_path, sess_options=so)
    sess.run(...)
    prof_file = sess.end_profiling()  # 返回 JSON 文件路径
    ```
  - NVTX: 在 C++ 编译选项打开 ENABLE_NVTX_PROFILE，或使用 nvtx 封装类把范围标注在关键 op/模块周围，配合 nsys 采集。
  - GEMM / kernel profiling：使用 ORT 的 gemm_profiler（在构建/运行阶段可启用）。
  - 可视化与解析：json chrome-trace、ORT 的 parse 脚本、nsys。
- 系统级 nsys（跨框架）
  - 命令示例（server）：
    ```
    nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o out --delay 30 --duration 60 \
      python -m <your_server_entry> --enable-layerwise-nvtx-marker
    ```
  - 命令示例（single batch CLI）：
    ```
    nsys profile -t cuda,nvtx -o myprofile python run_one_batch.py
    ```
  - 后处理：nsys export -t json out.nsys-rep 或在 Nsight UI 中打开。

4) 具体你应该收集的项（以便覆盖所有接口）
- 配置／元数据文件（JSON）
  - model id / path, model config (num_layers, d_model, etc.), batch size, input shapes
  - runtime env: CUDA/driver版本、GPU型号、PyTorch/ORT 版本、ONNXRuntime providers
  - profiling config: profiler type, activities, record_shapes, with_stack, profile_memory, warmup/active/wait, steps profiled, profile_prefix
- trace 文件（按 rank/worker 命名）
  - torch.profiler -> trace.json(.gz)
  - ORT -> profile_*.json
  - nsys -> .nsys-rep
  - pprof -> proto files（Go/C++服务）
- lightweight perf dumps（stage timing JSON）
  - SGLang 的 perf-dump、vLLM 的 summary stats、ORT 的 op timing解析输出
- 日志与 stdout/stderr（包含 profiler 启停日志）
- 辅助解析脚本与命令历史（方便重现）
- 如果分布式：每个节点的 hardware info + rank mapping 文件

5) 流程建议（步骤化）
1. 准备最小可复现用例（单输入/小 batch），记录 baseline 未开启 profiling 的延迟/吞吐。
2. 在代码里开启最细粒度的 marker（NVTX / record_function）并确保标注能唯一识别层/阶段。
3. 在配置中设置 schedule（wait/warmup/active）或短步数（num_steps=5）以限制 trace 大小。
4. 启动 nsys（若要 kernel 级），或用框架的 profiler（torch.profiler / ORT SessionOptions）触发采集。
5. 在采集结束后，先查看 high-level timing（stage-level JSON / perf dump），再打开 trace（perfetto / tensorboard / chrome://tracing）定位热点，再用 ncu/nsight compute 做 kernel 深入分析。
6. 若多卡：收集各 rank 的 trace 并合并（或在 perfetto 中并排对比）。
7. 记录所有环境与命令，保存 trace 与解析输出以便回溯。

6) 常见坑与注意事项
- profiling 会显著影响性能（不要在生产/benchmark 情况下全量打开）。
- trace 文件可能非常大：使用 schedule 或限制步数、或压缩（gzip）。
- CUPTI/kineto 依赖：某些 wheel 或自编译环境可能缺少，GPU trace 会失败或不完整。
- cuda-graph：若启用 cuda-graph，某些 profiler 可能捕捉不到内部 kernel 时间，需要额外配置（例如 nsys 的 cuda-graph-trace=node）。
- trace flush 可能耗时很久（大模型），需要提高 RPC/flush 超时（vLLM 文档提到 VLLM_RPC_TIMEOUT）。
- 堆栈追踪（with_stack）和 shape/memory profiling 增加开销：只在需要 debug 时开启。

7) 我可以为你做的后续工作（选项）
- 根据你具体用的框架与模型（发给我：框架名 + 启动命令或训练/推理脚本），我帮你生成一份 “一键 profiling 启动脚本”（包含 nsys / torch.profiler / ORT enable_profiling 的示例），并写好收集与合并命令。
- 帮你写 NVTX 或 torch.profiler 的最小标注补丁（把钩子/record_function 插到你的模型代码里）。
- 根据你上传的 trace（或给出文件路径）帮你解析并指出热点。

你现在想要哪个：我先帮你为某个具体框架（vLLM / SGLang / ORT / plain PyTorch）生成一份一键采样脚本，还是你把你要分析的模型/启动命令贴上来，我直接改成带 profiling 的启动命令？

结合上面的总结，给我改我的脚本，生成完整的profiling。要csv，json和可视化的结果

import torch
import time
import csv
import os
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
import numpy as np

# Configuration
MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
DEVICES = ["cpu", "cuda"]
DTYPES = [torch.bfloat16] # [torch.bfloat16, torch.float16, torch.float32]
NUM_WARMUP = 2
NUM_RUNS = 5
MAX_NEW_TOKENS = 500
OUTPUT_CSV = "smolvlm_benchmark_multi_dtype.csv"
OUTPUT_PLOT = "smolvlm_benchmark_multi_dtype.png"

def benchmark_smolvlm():
    # Load images once
    image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
    image2 = load_image("https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "Describe the images briefly."}
            ]
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    results = []

    for device_name in DEVICES:
        for dtype in DTYPES:
            display_name = "iGPU" if device_name == "cuda" else "CPU"
            dtype_str = str(dtype).split('.')[-1]
            
            print(f"\n--- Benchmarking {display_name} | {dtype_str} ---")
            
            device = torch.device(device_name)
            
            try:
                print(f"Loading model on {display_name} with {dtype_str}...")
                model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_ID,
                    torch_dtype=dtype,
                    _attn_implementation="eager",
                ).to(device)

                inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt").to(device)
                
                # Warmup
                print(f"Warming up ({NUM_WARMUP} run)...")
                for _ in range(NUM_WARMUP):
                    _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
                
                if device_name == "cuda":
                    torch.cuda.synchronize()

                # Benchmark
                print(f"Running benchmark ({NUM_RUNS} runs)...")
                latencies = []
                throughputs = [] 
                
                for i in range(NUM_RUNS):
                    start_time = time.time()
                    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
                    if device_name == "cuda":
                        torch.cuda.synchronize()
                    end_time = time.time()
                    
                    latency = end_time - start_time
                    num_tokens = output.shape[1] - inputs['input_ids'].shape[1]
                    tokens_per_sec = num_tokens / latency
                    
                    latencies.append(latency)
                    throughputs.append(tokens_per_sec)
                    print(f"  Run {i+1}: {latency:.2f}s, {tokens_per_sec:.2f} t/s")

                avg_latency = np.mean(latencies)
                avg_throughput = np.mean(throughputs)
                
                results.append({
                    "device": display_name,
                    "dtype": dtype_str,
                    "avg_latency_s": avg_latency,
                    "avg_tokens_per_sec": avg_throughput
                })
                
                # Clean up
                del model
                if device_name == "cuda":
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            except Exception as e:
                print(f"Error benchmarking {display_name} with {dtype_str}: {e}")
                results.append({
                    "device": display_name,
                    "dtype": dtype_str,
                    "avg_latency_s": 0,
                    "avg_tokens_per_sec": 0,
                    "error": str(e)
                })

    # Save to CSV
    keys = results[0].keys()
    with open(OUTPUT_CSV, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Generate Plot - Grouped Bar Chart
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Filter out failures
    df = df[df['avg_tokens_per_sec'] > 0]
    
    pivot_df = df.pivot(index='dtype', columns='device', values='avg_tokens_per_sec')
    
    ax = pivot_df.plot(kind='bar', figsize=(12, 7), rot=0, color=['skyblue', 'salmon'])
    
    plt.xlabel('Data Type')
    plt.ylabel('Throughput (Tokens/Second)')
    plt.title('SmolVLM Performance Comparison: CPU vs iGPU (Across Dtypes)')
    plt.legend(title="Device")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points')

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    benchmark_smolvlm()
```