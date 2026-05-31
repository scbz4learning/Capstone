import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

output_dir = Path('benchmark_charts')
output_dir.mkdir(exist_ok=True)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_latency(entry):
    lat = entry.get('latency', {})
    if 'ttft_ms_mean' in lat:
        return {'TTFT (ms)': lat['ttft_ms_mean'], 'TPOT (ms)': lat['tpot_ms_mean']}
    if 'latency_ms_mean' in lat:
        return {'Latency (ms)': lat['latency_ms_mean'], 'Throughput (img/s)': lat.get('batch_throughput_img_per_sec', 0)}
    return {}

def extract_power(entry):
    p = entry.get('power_energy', {})
    out = {'Avg Power (W)': p.get('avg_total_adjusted_w', 0), 'Energy per Inference (J)': p.get('energy_per_inference_j', 0)}
    if 'tokens_per_joule_p50' in p:
        out['Tokens per Joule'] = p['tokens_per_joule_p50']
    if 'img_per_sec_watt' in p and p.get('img_per_sec_watt', 0) > 0:
        out['Efficiency (img/W)'] = p['img_per_sec_watt']
    return out

def extract_memory(entry):
    m = entry.get('memory', {})
    if 'cpu_rss_mb' in m:
        return {'Peak Memory (GB)': m['cpu_rss_mb'] / 1024}
    return {'Peak Memory (GB)': m.get('peak_mem_allocated_gb', 0)}

def extract_metrics(entry):
    metrics = {}
    metrics.update(extract_latency(entry))
    metrics.update(extract_power(entry))
    metrics.update(extract_memory(entry))
    return metrics

smolvlm_pt_all = load_json('profiling_logs/smolvlm_pytorch.json')
vggt_all = load_json('profiling_logs/vggt_pytorch.json')
llamacpp_all = load_json('profiling_logs/smolvlm_llamacpp.json')

def load_llamacpp_data():
    records = {}
    for entry in llamacpp_all:
        if entry.get('status') != 'ok': continue
        model = entry.get('model', 'unknown')
        config = entry.get('config', '')
        if '_' in config:
            backend, quant = config.split('_', 1)
        else:
            backend, quant = config, 'unknown'
        records.setdefault((model, backend, quant), []).append(entry)
    return records

llamacpp_records = load_llamacpp_data()

def avg_metric(entries, metric_key):
    vals = [extract_metrics(e).get(metric_key) for e in entries if extract_metrics(e).get(metric_key, 0) > 0]
    return sum(vals) / len(vals) if vals else None

SMOLVLM_PT_CFGS = ['cuda_bfloat16_eager', 'cuda_bfloat16_sdpa', 'cpu_bfloat16_none']
VGGT_CFGS = ['cuda_float32_eager', 'cuda_float32_sdpa', 'cuda_bfloat16_eager', 'cuda_bfloat16_sdpa', 'cpu_float32_none', 'cpu_bfloat16_none']
LL_BACKENDS = ['cpu', 'vulkan', 'rocm']
LL_QUANTS = ['f16', 'Q8_0', 'Q4_K_M']

ENV_COL = {'Windows': '#1f77b4', 'WSL': '#ff7f0e', 'Linux': '#2ca02c', 'pt': '#1f77b4', 'lcpp': '#d62728'}
ENV_HATCH = {'PyTorch': '', 'llama.cpp': '///'}

def sanitize(s):
    return s.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_per_")

def plot_bars(ax, groups, group_labels, ylabel, title, legend_handles, fig_w=10, fig_h=4.5):
    max_bars = max(len(g) for g in groups) if groups else 1
    width = 0.85 / max_bars
    positions, vals = [], []
    for gi, group in enumerate(groups):
        gpos = gi + 0.1
        for bi, bar in enumerate(group):
            positions.append(gpos + bi * width)
            vals.append(bar['value'] if bar['value'] is not None else 0)

    colors = [bar.get('color', '#ccc') if bar['value'] is not None else '#ddd' for bar in sum(groups, [])]
    bars = ax.bar(positions, vals, width=width * 0.92, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    for bar, bdict in zip(bars, sum(groups, [])):
        bar.set_hatch(bdict.get('hatch', ''))
        if bdict['value'] is None:
            bar.set_alpha(0.25)
            yb = max(max(vals), 1) * 0.02 if vals else 0.02
            ax.text(bar.get_x() + bar.get_width() / 2., yb, 'N/T', ha='center', va='bottom', fontsize=4, color='gray')
        else:
            h = bar.get_height()
            if h < 0.001:    lbl = f'{h:.6f}'
            elif h < 1:      lbl = f'{h:.3f}'
            elif h < 100:    lbl = f'{h:.1f}'
            else:            lbl = f'{int(h)}'
            ax.text(bar.get_x() + bar.get_width() / 2., h, lbl, ha='center', va='bottom', fontsize=3.5)

    tick_pos = [i + 0.1 + (len(g) - 1) * width / 2 for i, g in enumerate(groups)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(group_labels, fontsize=6.5, rotation=0, ha='center')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig = ax.figure
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_h)
    fig.tight_layout(rect=[0, 0, 1.0, 1])
    if legend_handles:
        fig.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.0, 1), fontsize=6, ncol=1)

# ── 1. SmoVLM-Instruct ──
def plot_smolvlm_instruct():
    print("SmolVLM-Instruct chart...")
    pt_data = {}
    for e in smolvlm_pt_all:
        if e.get('status') != 'ok': continue
        pt_data[(e.get('_model'), e.get('config'), e.get('_environment'))] = e

    model = 'SmolVLM-Instruct'

    # Groups by Dtype: each group lists the Configuration names that form its bars
    # BF16: 9 bars (PyTorch, 3 envs × 3 attn modes)
    # f16/Q8_0/Q4_K_M: 3 bars each (Linux-llama.cpp × 3 backends)
    dtype_groups = {
        'BF16': [
            'Windows-PyTorch-CPU', 'WSL-PyTorch-CPU', 'Linux-PyTorch-CPU',
            'Windows-PyTorch-iGPU-Eager', 'WSL-PyTorch-iGPU-Eager', 'Linux-PyTorch-iGPU-Eager',
            'Windows-PyTorch-iGPU-SDPA', 'WSL-PyTorch-iGPU-SDPA', 'Linux-PyTorch-iGPU-SDPA',
        ],
        'f16': ['Linux-llama.cpp-CPU', 'Linux-llama.cpp-iGPU-Vulkan', 'Linux-llama.cpp-iGPU-ROCm'],
        'Q8_0': ['Linux-llama.cpp-CPU', 'Linux-llama.cpp-iGPU-Vulkan', 'Linux-llama.cpp-iGPU-ROCm'],
        'Q4_K_M': ['Linux-llama.cpp-CPU', 'Linux-llama.cpp-iGPU-Vulkan', 'Linux-llama.cpp-iGPU-ROCm'],
    }

    # Part 3 style: distinct color per device-framework, hatch per environment
    cfg_color = {
        'Windows-PyTorch-CPU': '#1f77b4', 'WSL-PyTorch-CPU': '#1f77b4', 'Linux-PyTorch-CPU': '#1f77b4',
        'Windows-PyTorch-iGPU-Eager': '#ff7f0e', 'WSL-PyTorch-iGPU-Eager': '#ff7f0e', 'Linux-PyTorch-iGPU-Eager': '#ff7f0e',
        'Windows-PyTorch-iGPU-SDPA': '#2ca02c', 'WSL-PyTorch-iGPU-SDPA': '#2ca02c', 'Linux-PyTorch-iGPU-SDPA': '#2ca02c',
        'Linux-llama.cpp-CPU': '#8c564b', 'Linux-llama.cpp-iGPU-Vulkan': '#9467bd', 'Linux-llama.cpp-iGPU-ROCm': '#e377c2',
    }
    cfg_hatch = {
        'Windows-PyTorch-CPU': '', 'WSL-PyTorch-CPU': '//', 'Linux-PyTorch-CPU': 'xx',
        'Windows-PyTorch-iGPU-Eager': '', 'WSL-PyTorch-iGPU-Eager': '//', 'Linux-PyTorch-iGPU-Eager': 'xx',
        'Windows-PyTorch-iGPU-SDPA': '', 'WSL-PyTorch-iGPU-SDPA': '//', 'Linux-PyTorch-iGPU-SDPA': 'xx',
        'Linux-llama.cpp-CPU': '', 'Linux-llama.cpp-iGPU-Vulkan': '', 'Linux-llama.cpp-iGPU-ROCm': '',
    }

    # Map config names to data sources
    def get_value(cfg_name, metric):
        # Parse config name -> (framework, env, device, backend)
        parts = cfg_name.split('-')
        fw = 'llama.cpp' if 'llama.cpp' in cfg_name else 'PyTorch'
        if fw == 'PyTorch':
            # Windows-PyTorch-CPU => env=Windows, cfg_name has device suffix
            env = parts[0]
            rest = '-'.join(parts[2:])  # e.g. 'CPU', 'iGPU-Eager', 'iGPU-SDPA'
            pt_cfg_map = {'CPU': 'cpu_bfloat16_none', 'iGPU-Eager': 'cuda_bfloat16_eager', 'iGPU-SDPA': 'cuda_bfloat16_sdpa'}
            pt_cfg = pt_cfg_map.get(rest)
            if not pt_cfg: return None
            entry = pt_data.get((model, pt_cfg, env))
            return extract_metrics(entry).get(metric) if entry else None
        else:
            # Linux-llama.cpp-CPU => bk=cpu; Linux-llama.cpp-iGPU-Vulkan => bk=vulkan
            bk = 'cpu' if 'CPU' in cfg_name else ('vulkan' if 'Vulkan' in cfg_name else 'rocm')
            # determine quant from the group key (passed separately)
            return None  # placeholder — filled in the loop below

    metrics = [('TTFT (ms)', 'ms'), ('TPOT (ms)', 'ms'), ('Avg Power (W)', 'W'),
               ('Energy per Inference (J)', 'J'), ('Peak Memory (GB)', 'GB'), ('Tokens per Joule', 'tokens/J')]

    quant_to_bk = {'f16': 'f16', 'Q8_0': 'Q8_0', 'Q4_K_M': 'Q4_K_M'}

    for metric, unit in metrics:
        fig, ax = plt.subplots()
        groups, labels = [], []
        for dtype_name, configs_in_group in dtype_groups.items():
            labels.append(dtype_name)
            bg = []
            for cfg_name in configs_in_group:
                v = None
                if 'PyTorch' in cfg_name:
                    parts = cfg_name.split('-')
                    env = parts[0]
                    rest = '-'.join(parts[2:])
                    pt_map = {'CPU': 'cpu_bfloat16_none', 'iGPU-Eager': 'cuda_bfloat16_eager', 'iGPU-SDPA': 'cuda_bfloat16_sdpa'}
                    pt_cfg = pt_map.get(rest)
                    if pt_cfg:
                        entry = pt_data.get((model, pt_cfg, env))
                        v = extract_metrics(entry).get(metric) if entry else None
                else:
                    bk = 'cpu' if '-CPU' in cfg_name else ('vulkan' if 'Vulkan' in cfg_name else 'rocm')
                    quant = dtype_name
                    entries = llamacpp_records.get((model, bk, quant))
                    v = avg_metric(entries, metric) if entries else None
                bg.append({
                    'value': v,
                    'color': cfg_color.get(cfg_name, '#ccc'),
                    'hatch': cfg_hatch.get(cfg_name, ''),
                    'label': cfg_name,
                })
            groups.append(bg)

        from matplotlib.patches import Patch
        all_cfgs = list(dict.fromkeys(sum(dtype_groups.values(), [])))  # unique, ordered
        leg = [Patch(facecolor=cfg_color[c], hatch=cfg_hatch.get(c, ''), label=c) for c in all_cfgs]
        plot_bars(ax, groups, labels, f'{metric} ({unit})', f'SmolVLM-Instruct: {metric}', leg, fig_w=12, fig_h=4)
        fname = f'smolvlm_instruct_{sanitize(metric)}.png'
        fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'  - {fname}')

# ── 2. VGGT ──
def plot_vggt():
    print("VGGT charts...")
    idx = {}
    for e in vggt_all:
        if e.get('status') != 'ok': continue
        cfg, env = e['config'], e.get('_environment')
        if env in ('Windows', 'WSL'):
            if cfg == 'cuda_float32': cfg = 'cuda_float32_eager'
            elif cfg == 'cuda_bfloat16': cfg = 'cuda_bfloat16_eager'
        idx[(cfg, env)] = e

    envs = ['Windows', 'WSL', 'Linux']
    cfg_disp = {'cuda_float32_eager': 'iGPU-F32-Eager', 'cuda_float32_sdpa': 'iGPU-F32-SDPA',
                'cuda_bfloat16_eager': 'iGPU-BF16-Eager', 'cuda_bfloat16_sdpa': 'iGPU-BF16-SDPA',
                'cpu_float32_none': 'CPU-F32', 'cpu_bfloat16_none': 'CPU-BF16'}

    metrics = [('Latency (ms)', 'ms'), ('Throughput (img/s)', 'img/s'), ('Avg Power (W)', 'W'),
               ('Energy per Inference (J)', 'J'), ('Peak Memory (GB)', 'GB'), ('Efficiency (mimg/W)', 'mimg/W')]

    # Part 3 style: distinct color per config, hatch per environment
    cfg_color_map = {
        'CPU-F32': '#1f77b4', 'CPU-BF16': '#ff7f0e',
        'iGPU-F32-Eager': '#2ca02c', 'iGPU-F32-SDPA': '#d62728',
        'iGPU-BF16-Eager': '#9467bd', 'iGPU-BF16-SDPA': '#8c564b',
    }
    env_hatch_map = {'Windows': '', 'WSL': '//', 'Linux': 'xx'}

    for metric, unit in metrics:
        fig, ax = plt.subplots()
        groups, labels = [], []
        for cfg in VGGT_CFGS:
            labels.append(cfg_disp.get(cfg, cfg))
            bg = []
            for env in envs:
                e = idx.get((cfg, env))
                v = None
                if e:
                    if 'Efficiency' in metric:
                        raw = extract_metrics(e).get('Efficiency (img/W)')
                        v = raw * 1000 if raw else None
                    else:
                        v = extract_metrics(e).get(metric)
                bg.append({
                    'value': v,
                    'color': cfg_color_map.get(cfg_disp.get(cfg, cfg), '#ccc'),
                    'hatch': env_hatch_map.get(env, ''),
                    'label': f'{env}-{cfg_disp.get(cfg, cfg)}',
                })
            groups.append(bg)

        from matplotlib.patches import Patch
        seen = {}
        for cfg in VGGT_CFGS:
            for env in envs:
                label = f'{env}-{cfg_disp.get(cfg, cfg)}'
                if label not in seen:
                    seen[label] = True
        leg = [Patch(facecolor=cfg_color_map.get(cfg_disp.get(cfg, cfg), '#ccc'),
                     hatch=env_hatch_map.get(env, ''),
                     label=f'{env}-{cfg_disp.get(cfg, cfg)}')
               for cfg in VGGT_CFGS for env in envs
               if f'{env}-{cfg_disp.get(cfg, cfg)}' in seen]
        plot_bars(ax, groups, labels, f'{metric} ({unit})', f'VGGT: {metric}', leg, fig_w=14, fig_h=4.5)
        fname_base = metric.replace(' (mimg/W)', '').replace(' (', '_').replace(')', '').replace('/', '_').lower()
        fname = f'vggt_{sanitize(fname_base)}.png'
        fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'  - {fname}')

# ⚠️ Fix: the efficiency chart had NaN-scale values causing matplotlib auto-scroll.  Using mimg/W (×1000)
# means values are 0.2-1.4, same order of magnitude as other metrics, so no blank space.

# ── 3. SmoVLM family (all models, PyTorch + llama.cpp) ──
def plot_family():
    print("SmolVLM family chart...")
    pt_data = {}
    for e in smolvlm_pt_all:
        if e.get('status') != 'ok': continue
        pt_data[(e.get('_model'), e.get('config'), e.get('_environment'))] = e

    # md table Section 3 columns: Model | Backend | Dtype
    # Model x-axis, Backend legend
    all_models = ['SmolVLM-256M-Instruct', 'SmolVLM-500M-Instruct', 'SmolVLM-Instruct',
                  'SmolVLM2-256M-Video-Instruct', 'SmolVLM2-500M-Video-Instruct', 'SmolVLM2-2.2B-Instruct']

    # Backend+Dtype combos to show per model
    backend_dtypes = [
        ('PyTorch-iGPU-sdpa', 'bf16'),
        ('llama.cpp-cpu', 'f16'), ('llama.cpp-cpu', 'Q8_0'), ('llama.cpp-cpu', 'Q4_K_M'),
        ('llama.cpp-vulkan', 'f16'), ('llama.cpp-vulkan', 'Q8_0'), ('llama.cpp-vulkan', 'Q4_K_M'),
        ('llama.cpp-rocm', 'f16'), ('llama.cpp-rocm', 'Q8_0'), ('llama.cpp-rocm', 'Q4_K_M'),
    ]

    # Color per backend+dtype using a map
    bar_colors = {
        'PyTorch-iGPU-sdpa': '#1f77b4',
        'llama.cpp-cpu-f16': '#8c564b', 'llama.cpp-cpu-Q8_0': '#8c564b', 'llama.cpp-cpu-Q4_K_M': '#8c564b',
        'llama.cpp-vulkan-f16': '#9467bd', 'llama.cpp-vulkan-Q8_0': '#9467bd', 'llama.cpp-vulkan-Q4_K_M': '#9467bd',
        'llama.cpp-rocm-f16': '#e377c2', 'llama.cpp-rocm-Q8_0': '#e377c2', 'llama.cpp-rocm-Q4_K_M': '#e377c2',
    }
    bar_hatches = {
        'PyTorch-iGPU-sdpa': '',
        'llama.cpp-cpu-f16': '', 'llama.cpp-cpu-Q8_0': '//', 'llama.cpp-cpu-Q4_K_M': 'xx',
        'llama.cpp-vulkan-f16': '', 'llama.cpp-vulkan-Q8_0': '//', 'llama.cpp-vulkan-Q4_K_M': 'xx',
        'llama.cpp-rocm-f16': '', 'llama.cpp-rocm-Q8_0': '//', 'llama.cpp-rocm-Q4_K_M': 'xx',
    }

    model_labels = {
        'SmolVLM-256M-Instruct': 'SmolVLM-256M-Instruct',
        'SmolVLM-500M-Instruct': 'SmolVLM-500M-Instruct',
        'SmolVLM-Instruct': 'SmolVLM-Instruct',
        'SmolVLM2-256M-Video-Instruct': 'SmolVLM2-256M-Video-Instruct',
        'SmolVLM2-500M-Video-Instruct': 'SmolVLM2-500M-Video-Instruct',
        'SmolVLM2-2.2B-Instruct': 'SmolVLM2-2.2B-Instruct',
    }

    metrics = [('TTFT (ms)', 'ms'), ('TPOT (ms)', 'ms'), ('Avg Power (W)', 'W'),
               ('Energy per Inference (J)', 'J'), ('Peak Memory (GB)', 'GB'), ('Tokens per Joule', 'tokens/J')]

    for metric, unit in metrics:
        fig, ax = plt.subplots()
        groups, labels = [], []
        for model in all_models:
            labels.append(model_labels.get(model, model))
            bg = []
            for bk, d in backend_dtypes:
                v = None
                if bk == 'PyTorch-iGPU-sdpa':
                    entry = pt_data.get((model, 'cuda_bfloat16_sdpa', 'Linux'))
                    v = extract_metrics(entry).get(metric) if entry else None
                else:
                    bk_short = bk.split('-')[1]  # 'cpu', 'vulkan', 'rocm'
                    entries = llamacpp_records.get((model, bk_short, d))
                    v = avg_metric(entries, metric) if entries else None
                key = f'{bk}-{d}'
                bg.append({'value': v, 'color': bar_colors.get(key, '#ccc'), 'hatch': bar_hatches.get(key, ''), 'label': key})
            groups.append(bg)

        from matplotlib.patches import Patch
        leg = []
        seen = set()
        for bk, d in backend_dtypes:
            key = f'{bk}-{d}'
            lbl = f'{bk} ({d})'
            if lbl not in seen:
                seen.add(lbl)
                leg.append(Patch(facecolor=bar_colors.get(key, '#ccc'), hatch=bar_hatches.get(key, ''), label=lbl))
        plot_bars(ax, groups, labels, f'{metric} ({unit})', f'SmolVLM family: {metric}', leg, fig_w=16, fig_h=5.5)
        fname = f'smolvlm_llamacpp_{sanitize(metric)}.png'
        fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'  - {fname}')

def main():
    plot_smolvlm_instruct()
    plot_vggt()
    plot_family()
    print(f'\nDone → {output_dir.absolute()}')

if __name__ == '__main__':
    main()
