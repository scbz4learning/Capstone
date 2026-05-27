import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data paths - relative to benckmarks directory
data_paths = {
    'smolvlm_windows': 'profiling_logs/smolvlm/run_20260526_111023/benchmarker_metrics_20260526_134336.json',
    'smolvlm_wsl': 'profiling_logs/smolvlm/run_20260526_155823_wsl/benchmarker_metrics_wsl_final_20260526_193214.json',
    'vggt_windows': 'profiling_logs/vggt/run_20260526_025409/benchmarker_metrics_20260526_064738.json',
    'vggt_wsl': 'profiling_logs/vggt/run_20260526_064741_wsl/benchmarker_metrics_wsl_final_20260526_111010.json',
}

def load_data(filepath):
    """Load JSON data from file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics(data):
    """Extract key metrics from benchmark data"""
    metrics = {}
    for entry in data:
        if entry['status'] != 'ok':
            continue
        
        config = entry['config']
        metrics[config] = {}
        
        # Extract latency metrics
        if 'latency' in entry:
            latency = entry['latency']
            if 'ttft_ms_mean' in latency:
                metrics[config]['TTFT (ms)'] = latency['ttft_ms_mean']
            if 'latency_ms_mean' in latency:
                metrics[config]['Latency (ms)'] = latency['latency_ms_mean']
            if 'tpot_ms_mean' in latency:
                metrics[config]['TPOT (ms)'] = latency['tpot_ms_mean']
            if 'batch_throughput_img_per_sec' in latency:
                metrics[config]['Throughput (img/s)'] = latency['batch_throughput_img_per_sec']
        
        # Extract power metrics
        if 'power_energy' in entry:
            power = entry['power_energy']
            metrics[config]['Avg Power (W)'] = power.get('avg_total_adjusted_w', 0)
            metrics[config]['Energy per Inference (J)'] = power.get('energy_per_inference_j', 0)
            if 'tokens_per_joule_p50' in power:
                metrics[config]['Tokens per Joule'] = power.get('tokens_per_joule_p50', 0)
            if 'img_per_sec_watt' in power and power.get('img_per_sec_watt', 0) > 0:
                metrics[config]['Efficiency (img/W)'] = power.get('img_per_sec_watt', 0)
        
        # Extract memory metrics - use total actual memory usage
        if 'memory' in entry:
            memory = entry['memory']
            # For CPU: use RSS memory (actual used), for GPU: use peak allocated
            if 'cpu_rss_mb' in memory:
                # CPU mode - convert RSS from MB to GB
                metrics[config]['Peak Memory (GB)'] = memory.get('cpu_rss_mb', 0) / 1024
            else:
                # GPU mode - use peak allocated memory (convert from MB to GB if needed, or use GB value)
                peak_mem = memory.get('peak_mem_allocated_gb', 0)
                if peak_mem < 1:  # If in MB (small value)
                    metrics[config]['Peak Memory (GB)'] = peak_mem / 1024
                else:  # Already in GB
                    metrics[config]['Peak Memory (GB)'] = peak_mem
    
    return metrics

def sanitize_filename(metric_key):
    """Sanitize metric key for use as filename"""
    # Replace special characters
    filename = metric_key.lower().replace(" ", "_")
    filename = filename.replace("(", "").replace(")", "")
    filename = filename.replace("/", "_per_")
    return filename

def format_config_name(config):
    """Format configuration name for display"""
    # Map config names to display names
    mapping = {
        'cpu_bfloat16_none': 'CPU',
        'cpu_float32_none': 'CPU',
        'cuda_bfloat16_eager': 'iGPU (eager)',
        'cuda_bfloat16_sdpa': 'iGPU (sdpa)',
        'cuda_float32': 'GPU'
    }
    return mapping.get(config, config)

def format_config_label(config, environment):
    """Format configuration label with environment"""
    # Map config names to display format with environment
    mapping = {
        'cpu_bfloat16_none': f'CPU ({environment})',
        'cpu_float32_none': f'CPU ({environment})',
        'cuda_bfloat16_eager': f'iGPU ({environment}, eager)',
        'cuda_bfloat16_sdpa': f'iGPU ({environment}, sdpa)',
        'cuda_float32': f'GPU ({environment})'
    }
    return mapping.get(config, f'{config} ({environment})')

def plot_comparison(model_name, windows_data, wsl_data, metric_key, unit=''):
    """Plot comparison chart for a specific metric"""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Extract metric name without unit from metric_key
    metric_name = metric_key.split('(')[0].strip()
    
    # Get all unique configs
    configs = sorted(set(list(windows_data.keys()) + list(wsl_data.keys())))
    
    # Define colors for each configuration
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    config_colors = {config: colors[i % len(colors)] for i, config in enumerate(configs)}
    
    # Create x-axis labels and positions
    labels = []
    positions = []
    values = []
    bar_colors = []
    
    pos = 0
    # For each config, add Windows data first, then WSL data
    for config in configs:
        # Windows data
        win_value = windows_data.get(config, {}).get(metric_key, 0)
        if win_value > 0:
            labels.append(format_config_label(config, 'Windows'))
            positions.append(pos)
            values.append(win_value)
            bar_colors.append(config_colors[config])
            pos += 1
        
        # WSL data
        wsl_value = wsl_data.get(config, {}).get(metric_key, 0)
        if wsl_value > 0:
            labels.append(format_config_label(config, 'WSL'))
            positions.append(pos)
            values.append(wsl_value)
            bar_colors.append(config_colors[config])
            pos += 1
    
    # Create bars
    bars = ax.bar(positions, values, width=0.7, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        # Use more decimal places for small values
        if height < 1:
            label = f'{height:.5f}'
        else:
            label = f'{height:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=8)
    
    # Customize plot
    ax.set_ylabel(f'{metric_name} ({unit})', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name}: {metric_name}', fontsize=13, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def main():
    # Load all data
    print("Loading benchmark data...")
    all_data = {key: load_data(path) for key, path in data_paths.items()}
    
    # Extract metrics
    print("Extracting metrics...")
    smolvlm_windows_metrics = extract_metrics(all_data['smolvlm_windows'])
    smolvlm_wsl_metrics = extract_metrics(all_data['smolvlm_wsl'])
    vggt_windows_metrics = extract_metrics(all_data['vggt_windows'])
    vggt_wsl_metrics = extract_metrics(all_data['vggt_wsl'])
    
    # Create output directory
    output_dir = Path('benchmark_charts')
    output_dir.mkdir(exist_ok=True)
    
    # SmoLVLM comparisons
    print("\nGenerating SmoLVLM comparison charts...")
    metrics_to_plot = [
        ('TTFT (ms)', 'ms'),
        ('TPOT (ms)', 'ms'),
        ('Avg Power (W)', 'W'),
        ('Energy per Inference (J)', 'J'),
        ('Peak Memory (GB)', 'GB'),
        ('Tokens per Joule', 'tokens/J')
    ]
    
    for metric, unit in metrics_to_plot:
        if any(metric in config_metrics for config_metrics in smolvlm_windows_metrics.values()):
            fig = plot_comparison('SmoLVLM', smolvlm_windows_metrics, smolvlm_wsl_metrics, metric, unit)
            filename = f'smolvlm_{sanitize_filename(metric)}.png'
            fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved: {filename}")
    
    # VGGT comparisons
    print("\nGenerating VGGT comparison charts...")
    vggt_metrics_to_plot = [
        ('Latency (ms)', 'ms'),
        ('Throughput (img/s)', 'images/second'),
        ('Avg Power (W)', 'W'),
        ('Energy per Inference (J)', 'J'),
        ('Peak Memory (GB)', 'GB')
    ]
    
    for metric, unit in vggt_metrics_to_plot:
        if any(metric in config_metrics for config_metrics in vggt_windows_metrics.values()):
            fig = plot_comparison('VGGT', vggt_windows_metrics, vggt_wsl_metrics, metric, unit)
            filename = f'vggt_{sanitize_filename(metric)}.png'
            fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved: {filename}")
    
    print(f"\nAll charts saved to: {output_dir.absolute()}")

if __name__ == '__main__':
    main()
