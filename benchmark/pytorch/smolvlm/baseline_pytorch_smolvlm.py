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
                    print(f"  Run {i+1}: {latency:.2f}s, {tokens_per_sec:.2f} t/s")

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
