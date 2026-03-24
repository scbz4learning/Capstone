import torch
import time
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Configuration
MODEL_ID = "facebook/VGGT-1B"
DEVICES = ["cpu", "cuda"]
DTYPES = [torch.bfloat16] # [torch.bfloat16, torch.float16, torch.float32]
NUM_WARMUP = 2
NUM_RUNS = 5
OUTPUT_CSV = "vggt_benchmark_multi_dtype.csv"
OUTPUT_PLOT = "vggt_benchmark_multi_dtype.png"

def benchmark_vggt():
    # Load and preprocess example images
    image_names = ["vggt/examples/kitchen/images/00.png", "vggt/examples/kitchen/images/01.png"]
    # Check if images exist, otherwise use placeholders if needed (assuming they exist as per test.py)
    if not all(os.path.exists(img) for img in image_names):
        print("Warning: Example images not found. Please ensure vggt/examples/kitchen/images/ exists.")
        return

    results = []

    for device_name in DEVICES:
        for dtype in DTYPES:
            display_name = "iGPU" if device_name == "cuda" else "CPU"
            dtype_str = str(dtype).split('.')[-1]
            
            print(f"\n--- Benchmarking {display_name} | {dtype_str} ---")
            
            device = torch.device(device_name)
            
            # Skip float16 on CPU as it is not well supported/optimized for many ops
            if device_name == "cpu" and dtype == torch.float16:
                print(f"Skipping float16 on CPU (not typically supported/efficient)")
                continue

            try:
                print(f"Loading VGGT model on {display_name} with {dtype_str}...")
                model = VGGT.from_pretrained(MODEL_ID).to(device)
                model.eval()

                # Preprocess images to device
                images = load_and_preprocess_images(image_names).to(device)

                # Warmup
                print(f"Warming up ({NUM_WARMUP} runs)...")
                with torch.no_grad():
                    # Note: VGGT test.py used autocast, we use explicit dtype via model.to(dtype) if possible, 
                    # but here we follow the pattern of test.py with autocast for consistency.
                    with torch.amp.autocast(device_type=device_name, dtype=dtype):
                        for _ in range(NUM_WARMUP):
                            _ = model(images)
                
                if device_name == "cuda":
                    torch.cuda.synchronize()

                # Benchmark
                print(f"Running benchmark ({NUM_RUNS} runs)...")
                latencies = []
                
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device_name, dtype=dtype):
                        for i in range(NUM_RUNS):
                            start_time = time.time()
                            _ = model(images)
                            if device_name == "cuda":
                                torch.cuda.synchronize()
                            end_time = time.time()
                            
                            latency = end_time - start_time
                            latencies.append(latency)
                            print(f"  Run {i+1}: {latency:.4f}s")

                avg_latency = np.mean(latencies)
                # For VGGT, throughput could be images per second
                avg_throughput = len(image_names) / avg_latency
                
                results.append({
                    "device": display_name,
                    "dtype": dtype_str,
                    "avg_latency_s": avg_latency,
                    "images_per_sec": avg_throughput
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
                    "images_per_sec": 0,
                    "error": str(e)
                })

    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nResults saved to {OUTPUT_CSV}")

        # Generate Plot
        df_plot = df[df['images_per_sec'] > 0]
        if not df_plot.empty:
            pivot_df = df_plot.pivot(index='dtype', columns='device', values='images_per_sec')
            ax = pivot_df.plot(kind='bar', figsize=(12, 7), rot=0, color=['skyblue', 'salmon'])
            
            plt.xlabel('Data Type')
            plt.ylabel('Throughput (Images/Second)')
            plt.title('VGGT Performance Comparison: CPU vs iGPU (Across Dtypes)')
            plt.legend(title="Device")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

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
    benchmark_vggt()
