import torch
import time
import json
import os
import subprocess
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import sys
import argparse
import glob
import datetime
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image

# Configuration
MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
DTYPES = [torch.bfloat16]
ATTN_IMPLS = ["eager"] 
NUM_WARMUP = 2
NUM_TEST = 5
MAX_NEW_TOKENS = 30
OUTPUT_DIR = "./profiling_logs/smolvlm"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class HardwareMonitor:
    def __init__(self, log_file):
        self.stop_event = threading.Event()
        self.log_file = log_file
        self.stats = []
        self.is_windows = os.name == 'nt'
        if not self.is_windows:
            self.thread = threading.Thread(target=self._monitor)
        
    def _monitor(self):
        cmd = ["rocm-smi", "--showpower", "--showuse", "--showclocks", "--json"]
        with open(self.log_file, "w", encoding='utf-8') as f:
            f.write("timestamp,power_w,sclk_mhz,mclk_mhz,gpu_use_pct\n")
            while not self.stop_event.is_set():
                try:
                    now = time.time()
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.stdout:
                        data = json.loads(result.stdout)
                        card_data = next(iter(data.values()))
                        power_str = str(card_data.get("Current Socket Graphics Package Power (W)", "0"))
                        sclk_str = str(card_data.get("sclk clock speed:", "0")).replace('(', '').replace('Mhz)', '').strip()
                        mclk_str = str(card_data.get("mclk clock speed:", "0")).replace('(', '').replace('Mhz)', '').strip()
                        
                        power_w = float(power_str) if power_str != "0" else 0.0
                        sclk_mhz = float(sclk_str) if sclk_str and sclk_str != "0" else 0.0
                        mclk_mhz = float(mclk_str) if mclk_str and mclk_str != "0" else 0.0
                        
                        f.write(f"{now},{power_w},{sclk_mhz},{mclk_mhz},0\n")
                        self.stats.append({
                            "timestamp": now,
                            "power_w": power_w,
                            "sclk_mhz": sclk_mhz,
                            "mclk_mhz": mclk_mhz
                        })
                except Exception as e: pass
                time.sleep(0.1)

    def start(self): 
        if not self.is_windows: self.thread.start()
    def stop(self):
        self.stop_event.set()
        if not self.is_windows: self.thread.join()
        return self.stats

def get_cpu_mem_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def get_valid_configs(args):
    configs = []
    devices = []
    if args.gpu: devices.append("cuda")
    if args.cpu: devices.append("cpu")
    
    for dev in devices:
        for dt in DTYPES:
            for attn in ATTN_IMPLS:
                if dev == "cpu" and attn == "flash_attention_2": continue
                if dt == torch.float32 and attn == "flash_attention_2": continue
                if dev == "cpu" and dt == torch.float16: continue
                configs.append({"device": dev, "dtype": dt, "attn": attn})
    return configs

def draw_smolvlm_plots(all_results):
    if not all_results: return
    df = pd.DataFrame(all_results)
    df["config"] = df["device"] + "\n" + df["dtype"] + "\n" + df["attn"]
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    metrics = [
        ("avg_ttft_ms", "TTFT (ms)", "skyblue"),
        ("avg_tpot_ms", "TPOT (ms)", "salmon"),
        ("peak_mem_gb", "Peak Memory (GB)", "lightgreen"),
        ("fps_per_watt", "Tokens/Sec per Watt", "gold"),
        ("energy_per_step_j", "Energy per Inference (J)", "orchid"),
        ("avg_power_w", "Avg Power (W)", "lightgrey")
    ]
    for i, (col, title, color) in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        if col in df.columns:
            ax.bar(df["config"], df[col], color=color)
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=45)
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "smolvlm_detailed_comparison.png"))
    print("Plots updated:", os.path.join(OUTPUT_DIR, "smolvlm_detailed_comparison.png"))

def profile_smolvlm(args):
    configs = get_valid_configs(args)
    if not configs:
        print("No valid configurations to run.")
        return

    is_windows = os.name == 'nt'
    needs_logging = is_windows and any(c["device"] == "cuda" for c in configs)

    print("--- Starting Advanced Profiling for SmolVLM ---")
    if needs_logging:
        while True:
            ans = input("[Windows] Please open 'AMD Software: Adrenalin Edition' -> 'Performance' -> 'Start Logging'. Have you started logging? (y/n): ")
            if ans.lower().strip() in ['y', 'yes', '']:
                break
        
    image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
    image2 = load_image("https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Describe the images briefly."}]}], add_generation_prompt=True)
    
    all_results = []

    for cfg in configs:
        dev_name = cfg["device"]
        dtype = cfg["dtype"]
        attn = cfg["attn"]
        dt_str = str(dtype).split(".")[-1]
        run_name = f"{dev_name}_{dt_str}_{attn}"
        print(f"\n--- Profiling Config: {run_name} ---")

        try:
            device = torch.device(dev_name)
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID, torch_dtype=dtype, attn_implementation=attn,
            ).to(device).eval()

            inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt").to(device)
            
            hw_log = os.path.join(OUTPUT_DIR, f"{run_name}_hw.csv")
            monitor = HardwareMonitor(hw_log)
            monitor.start()

            ttft_list, tpot_list = [], []
            cpu_mem_start = get_cpu_mem_gb()
            if dev_name == "cuda": torch.cuda.reset_peak_memory_stats()
            
            run_start_time = time.time()
            with torch.no_grad(), torch.amp.autocast(device_type=dev_name, dtype=dtype) if dtype != torch.float32 else torch.autocast(device_type=dev_name, enabled=False):
                for i in range(NUM_WARMUP + NUM_TEST):
                    torch.cuda.synchronize() if dev_name == "cuda" else None
                    t0 = time.perf_counter()
                    _ = model.generate(**inputs, max_new_tokens=1, use_cache=True)
                    torch.cuda.synchronize() if dev_name == "cuda" else None
                    ttft = (time.perf_counter() - t0) * 1000
                    
                    t_total_start = time.perf_counter()
                    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
                    torch.cuda.synchronize() if dev_name == "cuda" else None
                    total_time = (time.perf_counter() - t_total_start) * 1000
                    
                    tpot = (total_time - ttft) / (MAX_NEW_TOKENS - 1)
                    if i >= NUM_WARMUP:
                        ttft_list.append(ttft)
                        tpot_list.append(tpot)
            run_end_time = time.time()
                
            if dev_name == "cuda":
                peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            else:
                cpu_mem_end = get_cpu_mem_gb()
                peak_mem_gb = cpu_mem_end - cpu_mem_start
                if peak_mem_gb < 0.1: peak_mem_gb = cpu_mem_end

            hw_stats = monitor.stop()

            trace_file = os.path.join(OUTPUT_DIR, f"{run_name}_trace.json")
            print(f"Capturing Chrome Trace for {run_name}...")
            try:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA] if dev_name == "cuda" else [torch.profiler.ProfilerActivity.CPU],
                    record_shapes=True,
                    with_stack=False
                ) as prof:
                    with torch.no_grad(), torch.amp.autocast(device_type=dev_name, dtype=dtype) if dtype != torch.float32 else torch.autocast(device_type=dev_name, enabled=False):
                        model.generate(**inputs, max_new_tokens=2, use_cache=True)
                prof.export_chrome_trace(trace_file)
            except Exception as e:
                print(f"Trace capture failed: {e}")

            avg_power = np.mean([s['power_w'] for s in hw_stats]) if hw_stats else 0.0
            avg_latency_s = (np.mean(ttft_list) + np.mean(tpot_list) * (MAX_NEW_TOKENS-1)) / 1000
            
            tps = MAX_NEW_TOKENS / avg_latency_s
            fps_watt = tps / avg_power if avg_power > 0 else 0.0
            energy_joules = avg_power * avg_latency_s

            res = {
                "device": dev_name, "dtype": dt_str, "attn": attn,
                "avg_ttft_ms": np.mean(ttft_list), "avg_tpot_ms": np.mean(tpot_list),
                "peak_mem_gb": peak_mem_gb,
                "avg_power_w": avg_power,
                "fps_per_watt": fps_watt,
                "energy_per_step_j": energy_joules,
                "clocks_sclk_avg": np.mean([s['sclk_mhz'] for s in hw_stats]) if hw_stats else 0,
                "hw_log_file": hw_log,
                "run_start_time": run_start_time,
                "run_end_time": run_end_time,
                "config": run_name
            }
            all_results.append(res)
            print(f"Results: TTFT {res['avg_ttft_ms']:.1f}ms | TPOT {res['avg_tpot_ms']:.1f}ms | Mem {res['peak_mem_gb']:.2f}GB")

            del model
            torch.cuda.empty_cache() if dev_name == "cuda" else None

        except Exception as e:
            print(f"Failed config {run_name}: {e}")

    with open(os.path.join(OUTPUT_DIR, "smolvlm_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=4)
        
    draw_smolvlm_plots(all_results)
    
    if needs_logging:
        while True:
            ans = input("\n[Windows] Profiling finished. Please stop logging in Adrenalin now. Have you stopped logging? (y/n): ")
            if ans.lower().strip() in ['y', 'yes', '']:
                break
        post_process()

def post_process():
    metrics_file = os.path.join(OUTPUT_DIR, "smolvlm_metrics.json")
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}. Please run profiling first.")
        return
    with open(metrics_file, "r") as f:
        results = json.load(f)
    
    cn_dir = os.path.join(os.getenv('LOCALAPPDATA', ''), 'AMD', 'CN')
    csv_files = glob.glob(os.path.join(cn_dir, "Hardware.*.CSV"))
    csv_files.sort(key=os.path.getmtime, reverse=True)
    if not csv_files:
        print(f"No CSV files found in {cn_dir}. Please export logs from Adrenalin.")
        return
    
    print(f"Found {len(csv_files)} Hardware CSVs in {cn_dir}, reading the latest: {csv_files[0]}")

    def parse_time(ts_str):
        try: return datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()
        except: return 0.0

    for res in results:
        r_start = res.get("run_start_time", 0)
        r_end = res.get("run_end_time", 0)
        if r_start == 0: continue
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df = df[df["TIME STAMP"] != "N/A"]
                df["ts"] = df["TIME STAMP"].apply(parse_time)
                mask = (df["ts"] >= (r_start - 2.0)) & (df["ts"] <= (r_end + 2.0))
                matched = df[mask]
                if not matched.empty:
                    avg_pwr = float(matched["GPU PWR"].mean())
                    res["avg_power_w"] = avg_pwr
                    res["clocks_sclk_avg"] = float(matched["GPU SCLK"].mean())
                    print(f"[{res['config']}] Matched {len(matched)} CSV records, avg power: {avg_pwr:.1f}W")
                    if "avg_ttft_ms" in res:
                        avg_latency_s = (res["avg_ttft_ms"] + res["avg_tpot_ms"] * (MAX_NEW_TOKENS-1)) / 1000
                        tps = MAX_NEW_TOKENS / avg_latency_s
                        res["fps_per_watt"] = tps / avg_pwr if avg_pwr > 0 else 0
                        res["energy_per_step_j"] = avg_pwr * avg_latency_s
                    break
            except Exception as e:
                print(f"Failed to read {csv_file}: {e}")

    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=4)
    draw_smolvlm_plots(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile SmolVLM model.")
    parser.add_argument("--cpu", action="store_true", help="Enable CPU profiling")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU profiling")
    parser.add_argument("--npu", action="store_true", help="Enable NPU profiling (not fully supported yet)")
    args = parser.parse_args()
    
    if not (args.cpu or args.gpu or args.npu):
        args.cpu = args.gpu = args.npu = True
        
    profile_smolvlm(args)
