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

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third-party"))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Configuration
MODEL_ID = "facebook/VGGT-1B"
DEVICES = ["cuda", "cpu"]
DTYPES = [torch.bfloat16]
NUM_WARMUP = 2
NUM_TEST = 5
OUTPUT_DIR = "./profiling_logs/vggt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class HardwareMonitor:
    def __init__(self, log_file):
        self.stop_event = threading.Event()
        self.log_file = log_file
        self.stats = []
        self.thread = threading.Thread(target=self._monitor)
        
    def _monitor(self):
        cmd = ["rocm-smi", "--showpower", "--showuse", "--showclocks", "--json"]
        with open(self.log_file, "w") as f:
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

    def start(self): self.thread.start()
    def stop(self):
        self.stop_event.set()
        self.thread.join()
        return self.stats

def get_cpu_mem_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def get_valid_configs():
    configs = []
    for dev in DEVICES:
        for dt in DTYPES:
            if dev == "cpu" and dt == torch.float16: continue
            configs.append({"device": dev, "dtype": dt})
    return configs

def profile_vggt():
    print("--- Starting Advanced Profiling for VGGT ---")
    image_names = ["third-party/vggt/examples/kitchen/images/00.png", "third-party/vggt/examples/kitchen/images/01.png"]
    if not all(os.path.exists(img) for img in image_names):
        print(f"Warning: Example images not found. Looked for {image_names}")
        return
        
    configs = get_valid_configs()
    all_results = []

    for cfg in configs:
        dev_name = cfg["device"]
        dtype = cfg["dtype"]
        dt_str = str(dtype).split(".")[-1]
        run_name = f"{dev_name}_{dt_str}"
        print(f"\n--- Profiling Config: {run_name} ---")

        try:
            device = torch.device(dev_name)
            # Fix initialization: VGGT doesn't accept torch_dtype or attn_implementation
            # Do NOT use .to(dtype) on the model itself, as VGGT heads expect Float due to manual autocast(False) in its code
            model = VGGT.from_pretrained(MODEL_ID).to(device).eval()
            images = load_and_preprocess_images(image_names).to(device)
            # VGGT expects images of shape [1, S, 3, H, W] for sequence, wrapper handles it
            
            hw_log = os.path.join(OUTPUT_DIR, f"{run_name}_hw.csv")
            monitor = HardwareMonitor(hw_log)
            monitor.start()

            latencies = []
            
            cpu_mem_start = get_cpu_mem_gb()
            if dev_name == "cuda":
                torch.cuda.reset_peak_memory_stats()
                
            with torch.no_grad(), torch.amp.autocast(device_type=dev_name, dtype=dtype) if dtype != torch.float32 else torch.autocast(device_type=dev_name, enabled=False):
                # 1 image pair processed per call (2 frames total?) Actually load_and_preprocess_images loads them into a video sequence tensor.
                for i in range(NUM_WARMUP + NUM_TEST):
                    torch.cuda.synchronize() if dev_name == "cuda" else None
                    t0 = time.perf_counter()
                    
                    # Single forward pass for Vision Model
                    outputs = model(images)
                    
                    torch.cuda.synchronize() if dev_name == "cuda" else None
                    latency = (time.perf_counter() - t0) * 1000
                    
                    if i >= NUM_WARMUP:
                        latencies.append(latency)
                
            if dev_name == "cuda":
                peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            else:
                cpu_mem_end = get_cpu_mem_gb()
                peak_mem_gb = cpu_mem_end - cpu_mem_start
                if peak_mem_gb < 0.1: peak_mem_gb = cpu_mem_end

            hw_stats = monitor.stop()
            avg_power = np.mean([s['power_w'] for s in hw_stats]) if hw_stats else 0.0
            avg_latency_s = np.mean(latencies) / 1000
            
            num_frames = images.shape[1] if len(images.shape) == 5 else 1
            fps = num_frames / avg_latency_s
            fps_watt = fps / avg_power if avg_power > 0 else 0.0
            energy_joules = avg_power * avg_latency_s

            res = {
                "device": dev_name, "dtype": dt_str,
                "avg_latency_ms": np.mean(latencies),
                "peak_mem_gb": peak_mem_gb,
                "avg_power_w": avg_power,
                "fps_per_watt": fps_watt,
                "energy_per_step_j": energy_joules,
                "hw_log_file": hw_log
            }
            all_results.append(res)
            print(f"Results: Latency {res['avg_latency_ms']:.1f}ms | Mem {res['peak_mem_gb']:.2f}GB | Power {res['avg_power_w']:.1f}W | FPS {fps:.1f}")

            del model
            torch.cuda.empty_cache() if dev_name == "cuda" else None
        except Exception as e: print(f"Error: {e}")

    with open(os.path.join(OUTPUT_DIR, "vggt_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=4)
        
    if all_results:
        df = pd.DataFrame(all_results)
        df["config"] = df["device"] + "\n" + df["dtype"]
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        metrics = [
            ("avg_latency_ms", "Latency (ms)", "skyblue"),
            ("peak_mem_gb", "Peak Memory (GB)", "lightgreen"),
            ("fps_per_watt", "FPS per Watt", "gold"),
            ("energy_per_step_j", "Energy per Inference (J)", "orchid"),
        ]
        for i, (col, title, color) in enumerate(metrics):
            ax = axes[i]
            ax.bar(df["config"], df[col], color=color)
            ax.set_title(title)
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', xytext=(0, 5), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "vggt_detailed_comparison.png"))
        print("\nProfiling Complete! Data saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    profile_vggt()
