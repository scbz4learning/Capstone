import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from matplotlib.patches import Patch

output_dir = Path('benchmark_charts')
output_dir.mkdir(exist_ok=True)

METRIC_DIRECTION = {
    'TTFT (ms)': 'lower is better',
    'TPOT (ms)': 'lower is better',
    'Latency (ms)': 'lower is better',
    'Avg Power (W)': 'lower is better',
    'Energy per Inference (J)': 'lower is better',
    'Peak Memory (GB)': 'lower is better',
    'Throughput (img/s)': 'higher is better',
    'Tokens per Joule': 'higher is better',
    'Efficiency (img/W)': 'higher is better',
    'Efficiency (million images / W)': 'higher is better',
}

def make_footnote(metric):
    direction = METRIC_DIRECTION.get(metric)
    base = "* fell back to CPU"
    if direction:
        return f"{base}\nDirection: {direction}"
    return base

# Rainbow colors: Red > Orange > Yellow > Green > Cyan > Blue > Purple
# Same color = same execution target (across all charts)
CFG_COLORS_BY_SEMANTICS = {
    # PyTorch CPU (all envs)
    'Windows-PyTorch-CPU': '#d62728',
    'WSL-PyTorch-CPU': '#d62728',
    'Linux-PyTorch-CPU': '#d62728',
    # PyTorch iGPU-Eager (all envs)
    'Windows-PyTorch-iGPU-Eager': '#ff7f0e',
    'WSL-PyTorch-iGPU-Eager': '#ff7f0e',
    'Linux-PyTorch-iGPU-Eager': '#ff7f0e',
    # PyTorch iGPU-SDPA (all envs)
    'Windows-PyTorch-iGPU-SDPA': '#bcbd22',
    'WSL-PyTorch-iGPU-SDPA': '#bcbd22',
    'Linux-PyTorch-iGPU-SDPA': '#bcbd22',
    # llama.cpp CPU
    'Linux-llama.cpp-CPU': '#2ca02c',
    # llama.cpp Vulkan (real GPU)
    'Linux-llama.cpp-iGPU-Vulkan': '#17becf',
    # llama.cpp ROCm
    'Linux-llama.cpp-iGPU-ROCm': '#1f77b4',
}

# VGGT semantic colors (matching rainbow scheme)
VGGT_CFG_COLORS_BY_SEMANTICS = {
    'CPU-F32': '#d62728',        # Red
    'CPU-BF16': '#d62728',       # Red (CPU)
    'iGPU-F32-Eager': '#ff7f0e', # Orange (PyTorch iGPU eager / BF16)
    'iGPU-F32-SDPA': '#bcbd22',  # Yellow (PyTorch iGPU SDPA)
    'iGPU-BF16-Eager': '#ff7f0e', # Orange (PyTorch iGPU eager / BF16)
    'iGPU-BF16-SDPA': '#bcbd22',  # Yellow (same as iGPU-F32-SDPA)
}

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_effective_device(entry):
    """Determine if data was actually collected on GPU or CPU fallback.

    - Vulkan (llama.cpp): always real GPU execution -> 'gpu'
    - ROCm: if gpu_vram < 200MB, the model ran on CPU (ROCm fallback) -> 'cpu'
    - PyTorch cuda on integrated: if gpu_vram < 200MB -> 'cpu'
    - Otherwise -> 'cpu'
    """
    config = entry.get('config', '')
    backend = config.split('_', 1)[0] if '_' in config else config

    if backend == 'vulkan':
        return 'gpu'

    m = entry.get('memory', {})
    gpu_vram_mb = m.get('gpu_vram_peak_mb', 0)

    if backend == 'rocm':
        if gpu_vram_mb < 200:
            return 'cpu'
        return 'gpu'

    is_integrated = entry.get('is_integrated', False)
    if is_integrated:
        if gpu_vram_mb < 200:
            return 'cpu'
        return 'gpu'

    return 'cpu'

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
    peak_gb = m.get('peak_mem_allocated_gb', 0)
    if peak_gb > 0:
        return {'Peak Memory (GB)': peak_gb}
    return {'Peak Memory (GB)': m.get('cpu_rss_mb', 0) / 1024}

def extract_metrics(entry):
    metrics = {}
    metrics.update(extract_latency(entry))
    metrics.update(extract_power(entry))
    metrics.update(extract_memory(entry))
    return metrics

def extract_metrics_with_error(entry):
    lat = entry.get('latency', {})
    metrics = {}
    errors = {}
    if 'ttft_ms_mean' in lat:
        metrics['TTFT (ms)'] = lat['ttft_ms_mean']
        errors['TTFT (ms)'] = abs(lat.get('ttft_ms_p95', lat['ttft_ms_mean']) - lat['ttft_ms_mean'])
        metrics['TPOT (ms)'] = lat['tpot_ms_mean']
        errors['TPOT (ms)'] = abs(lat.get('tpot_ms_p95', lat['tpot_ms_mean']) - lat['tpot_ms_mean'])
    if 'latency_ms_mean' in lat:
        metrics['Latency (ms)'] = lat['latency_ms_mean']
        errors['Latency (ms)'] = abs(lat.get('latency_ms_p95', lat['latency_ms_mean']) - lat['latency_ms_mean'])
        metrics['Throughput (img/s)'] = lat.get('batch_throughput_img_per_sec', 0)
    p = entry.get('power_energy', {})
    metrics['Avg Power (W)'] = p.get('avg_total_adjusted_w', 0)
    metrics['Energy per Inference (J)'] = p.get('energy_per_inference_j', 0)
    if 'tokens_per_joule_p50' in p:
        metrics['Tokens per Joule'] = p['tokens_per_joule_p50']
    if 'img_per_sec_watt' in p and p.get('img_per_sec_watt', 0) > 0:
        metrics['Efficiency (img/W)'] = p['img_per_sec_watt']
    m = entry.get('memory', {})
    peak_gb = m.get('peak_mem_allocated_gb', 0)
    if peak_gb > 0:
        metrics['Peak Memory (GB)'] = peak_gb
    else:
        metrics['Peak Memory (GB)'] = m.get('cpu_rss_mb', 0) / 1024
    return metrics, errors

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
    vals = []
    errs = []
    is_fallback = False
    for e in entries:
        m, err = extract_metrics_with_error(e)
        v = m.get(metric_key)
        if v is not None and v > 0:
            vals.append(v)
            errs.append(err.get(metric_key) if err.get(metric_key) is not None else 0)
        if not is_fallback and get_effective_device(e) == 'cpu':
            is_fallback = True
    if not vals:
        return None, None, False
    avg = sum(vals) / len(vals)
    err = sum(errs) / len(errs) if errs else None
    return avg, err, is_fallback

SMOLVLM_PT_CFGS = ['cuda_bfloat16_eager', 'cuda_bfloat16_sdpa', 'cpu_bfloat16_none']
VGGT_CFGS = ['cuda_float32_eager', 'cuda_float32_sdpa', 'cpu_float32_none',
             'cuda_bfloat16_eager', 'cuda_bfloat16_sdpa', 'cpu_bfloat16_none']
LL_BACKENDS = ['cpu', 'vulkan', 'rocm']
LL_QUANTS = ['f16', 'Q8_0', 'Q4_K_M']

ENV_COL = {'Windows': '#1f77b4', 'WSL': '#ff7f0e', 'Linux': '#2ca02c', 'pt': '#1f77b4', 'lcpp': '#d62728'}
ENV_HATCH = {'PyTorch': '', 'llama.cpp': '///'}

def sanitize(s):
    return s.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_per_")

def plot_bars(ax, groups, group_labels, ylabel, title, legend_handles, fig_w=10, fig_h=4.5, errors=None, footnote=None, group_start=None):
    max_bars = max(len(g) for g in groups) if groups else 1
    width = 0.85 / max_bars
    positions, vals = [], []
    for gi, group in enumerate(groups):
        gpos = group_start[gi] if group_start and gi < len(group_start) else (gi + 0.1)
        for bi, bar in enumerate(group):
            positions.append(gpos + bi * width)
            vals.append(bar['value'] if bar['value'] is not None else 0)

    flat_bars = sum(groups, [])
    colors = [bar.get('color', '#ccc') if bar['value'] is not None else '#ddd' for bar in flat_bars]

    bars = ax.bar(positions, vals, width=width * 0.92, color=colors)
    for bar, bdict in zip(bars, flat_bars):
        if 'hatch' in bdict:
            bar.set_hatch(bdict['hatch'])
        if 'F32' in bdict.get('label', ''):
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)
        else:
            bar.set_edgecolor('black')
            bar.set_linewidth(0.5)

    if errors and any(e is not None for e in errors):
        for i, (bar, err) in enumerate(zip(bars, errors)):
            if err is not None:
                ax.errorbar(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    yerr=err,
                    fmt='none',
                    ecolor='black',
                    capsize=2,
                    linewidth=0.8,
                )

    for bar, bdict in zip(bars, flat_bars):
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

    tick_pos = [(group_start[gi] if group_start and gi < len(group_start) else (gi + 0.1)) + (len(g) - 1) * width / 2 for gi, g in enumerate(groups)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(group_labels, fontsize=6.5, rotation=0, ha='center')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig = ax.figure
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_h)
    fig.tight_layout()

    pos = ax.get_position()
    ax_r = pos.x0 + pos.width
    ax_t = pos.y0 + pos.height
    ax_b = pos.y0

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(ax_r + 0.015, ax_t),
            frameon=True,
            fontsize=6,
        )

    if footnote:
        fig.text(
            x=ax_r + 0.01,
            y=ax_b,
            s=footnote,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox=dict(facecolor='#f5f5f5', edgecolor='#cccccc', boxstyle='round,pad=0.3'),
        )

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
    cfg_color = CFG_COLORS_BY_SEMANTICS
    cfg_hatch = {
        'Windows-PyTorch-CPU': 'xx', 'WSL-PyTorch-CPU': '//', 'Linux-PyTorch-CPU': '',
        'Windows-PyTorch-iGPU-Eager': 'xx', 'WSL-PyTorch-iGPU-Eager': '//', 'Linux-PyTorch-iGPU-Eager': '',
        'Windows-PyTorch-iGPU-SDPA': 'xx', 'WSL-PyTorch-iGPU-SDPA': '//', 'Linux-PyTorch-iGPU-SDPA': '',
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
                err = None
                fallback = False
                if 'PyTorch' in cfg_name:
                    parts = cfg_name.split('-')
                    env = parts[0]
                    rest = '-'.join(parts[2:])
                    pt_map = {'CPU': 'cpu_bfloat16_none', 'iGPU-Eager': 'cuda_bfloat16_eager', 'iGPU-SDPA': 'cuda_bfloat16_sdpa'}
                    pt_cfg = pt_map.get(rest)
                    if pt_cfg:
                        entry = pt_data.get((model, pt_cfg, env))
                        if entry:
                            m, e = extract_metrics_with_error(entry)
                            v = m.get(metric)
                            err = e.get(metric)
                            if metric == 'Peak Memory (GB)' and pt_cfg == 'cpu_bfloat16_none':
                                v = entry.get('memory', {}).get('cpu_rss_mb', 0) / 1024
                else:
                    bk = 'cpu' if '-CPU' in cfg_name else ('vulkan' if 'Vulkan' in cfg_name else 'rocm')
                    quant = dtype_name
                    entries = llamacpp_records.get((model, bk, quant))
                    v, err, fallback = avg_metric(entries, metric) if entries else (None, None, False)
                bg.append({
                    'value': v,
                    'error': err,
                    'color': cfg_color.get(cfg_name, '#ccc'),
                    'hatch': cfg_hatch.get(cfg_name, ''),
                    'label': cfg_name + (' *' if fallback else ''),
                })
            groups.append(bg)

        all_cfgs = list(dict.fromkeys(sum(dtype_groups.values(), [])))

        rocm_fallback = False
        for bk, quant_list in [('rocm', ['f16', 'Q8_0', 'Q4_K_M'])]:
            for q in quant_list:
                entries = llamacpp_records.get((model, bk, q))
                if entries and any(get_effective_device(e) == 'cpu' for e in entries):
                    rocm_fallback = True
                    break

        color_legend = [
            Patch(facecolor='#d62728', label='PyTorch-CPU'),
            Patch(facecolor='#ff7f0e', label='PyTorch iGPU-Eager'),
            Patch(facecolor='#bcbd22', label='PyTorch iGPU-SDPA'),
            Patch(facecolor='#2ca02c', label='llama.cpp-CPU'),
            Patch(facecolor='#17becf', label='llama.cpp-Vulkan'),
            Patch(facecolor='#1f77b4', label='llama.cpp-ROCm' + (' *' if rocm_fallback else '')),
        ]
        hatch_legend = [
            Patch(facecolor='white', edgecolor='gray', hatch='xx', label='Windows'),
            Patch(facecolor='white', edgecolor='gray', hatch='//', label='WSL'),
            Patch(facecolor='white', edgecolor='gray', hatch='', label='Linux'),
        ]
        leg = color_legend + [Patch(facecolor='none', edgecolor='none', label='')] + hatch_legend

        err_vals = []
        for g in groups:
            for b in g:
                if b['value'] is not None and b.get('error') is not None:
                    err_vals.append(b['error'])
                else:
                    err_vals.append(None)

        group_starts = [0.1, 1.1, 1.6, 2.1]
        plot_bars(ax, groups, labels, f'{metric} ({unit})', f'SmolVLM-Instruct: {metric}', leg, fig_w=12, fig_h=4, errors=err_vals if any(v is not None for v in err_vals) else None, footnote=make_footnote(metric), group_start=group_starts)
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
               ('Energy per Inference (J)', 'J'), ('Peak Memory (GB)', 'GB'), ('Efficiency (million images / W)', 'million images / W')]

    # Rainbow colors: Red > Orange > Yellow > Green > Cyan > Blue > Purple
    cfg_color_map = VGGT_CFG_COLORS_BY_SEMANTICS
    env_hatch_map = {'Windows': 'xx', 'WSL': '//', 'Linux': ''}

    for metric, unit in metrics:
        fig, ax = plt.subplots()
        groups, labels = [], []
        for precision_name, cfgs in [('F32', ['cuda_float32_eager', 'cuda_float32_sdpa', 'cpu_float32_none']),
                                      ('BF16', ['cuda_bfloat16_eager', 'cuda_bfloat16_sdpa', 'cpu_bfloat16_none'])]:
            labels.append(precision_name)
            bg = []
            for cfg in cfgs:
                for env in envs:
                    e = idx.get((cfg, env))
                    v = None
                    err = None
                    if e:
                        if 'Efficiency' in metric:
                            m = extract_metrics(e)
                            raw = m.get('Efficiency (img/W)')
                            v = raw * 1000 if raw else None
                        else:
                            m, er = extract_metrics_with_error(e)
                            v = m.get(metric)
                            err = er.get(metric)
                            if 'Peak Memory' in metric and cfg.startswith('cpu_'):
                                mem = e.get('memory', {})
                                v = mem.get('cpu_rss_mb', 0) / 1024
                    bg.append({
                        'value': v,
                        'error': err,
                        'color': cfg_color_map.get(cfg_disp.get(cfg, cfg), '#ccc'),
                        'hatch': env_hatch_map.get(env, ''),
                        'label': f'{env}-{cfg_disp.get(cfg, cfg)}',
                    })
            groups.append(bg)

        seen = {}
        for cfg in VGGT_CFGS:
            for env in envs:
                label = f'{env}-{cfg_disp.get(cfg, cfg)}'
                if label not in seen:
                    seen[label] = True

        color_legend = [
            Patch(facecolor=cfg_color_map.get('CPU-F32', '#ccc'), hatch='', label='CPU'),
            Patch(facecolor=cfg_color_map.get('iGPU-F32-Eager', '#ccc'), hatch='', label='iGPU-Eager'),
            Patch(facecolor=cfg_color_map.get('iGPU-F32-SDPA', '#ccc'), hatch='', label='iGPU-SDPA'),
        ]

        hatch_legend = [
            Patch(facecolor='white', edgecolor='gray', hatch='xx', label='Windows'),
            Patch(facecolor='white', edgecolor='gray', hatch='//', label='WSL'),
            Patch(facecolor='white', edgecolor='gray', hatch='', label='Linux'),
        ]
        alpha_legend = [
            Patch(facecolor='white', edgecolor='black', linewidth=2, label='F32'),
            Patch(facecolor='white', edgecolor='black', linewidth=0.5, label='BF16'),
        ]
        leg = color_legend + [Patch(facecolor='none', edgecolor='none', label='')] + hatch_legend + [Patch(facecolor='none', edgecolor='none', label='')] + alpha_legend

        err_vals = []
        for g in groups:
            for b in g:
                if b['value'] is not None and b.get('error') is not None:
                    err_vals.append(b['error'])
                else:
                    err_vals.append(None)

        plot_bars(ax, groups, labels, f'{metric} ({unit})', f'VGGT: {metric}', leg, fig_w=14, fig_h=4.5, errors=err_vals if any(v is not None for v in err_vals) else None, footnote=make_footnote(metric))
        fname_base = metric.replace(' (million images / W)', '').replace(' (', '_').replace(')', '').replace('/', '_').lower()
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
        ('llama.cpp-cpu', 'f16'), ('llama.cpp-vulkan', 'f16'), ('llama.cpp-rocm', 'f16'),
        ('llama.cpp-cpu', 'Q8_0'), ('llama.cpp-vulkan', 'Q8_0'), ('llama.cpp-rocm', 'Q8_0'),
        ('llama.cpp-cpu', 'Q4_K_M'), ('llama.cpp-vulkan', 'Q4_K_M'), ('llama.cpp-rocm', 'Q4_K_M'),
    ]

    # Color per backend+dtype using a map
    bar_colors = {
        'PyTorch-iGPU-sdpa': '#bcbd22',
        'llama.cpp-cpu-f16': '#2ca02c', 'llama.cpp-cpu-Q8_0': '#2ca02c', 'llama.cpp-cpu-Q4_K_M': '#2ca02c',
        'llama.cpp-vulkan-f16': '#17becf', 'llama.cpp-vulkan-Q8_0': '#17becf', 'llama.cpp-vulkan-Q4_K_M': '#17becf',
        'llama.cpp-rocm-f16': '#1f77b4', 'llama.cpp-rocm-Q8_0': '#1f77b4', 'llama.cpp-rocm-Q4_K_M': '#1f77b4',
    }
    bar_hatches = {
        'PyTorch-iGPU-sdpa': '',
        'llama.cpp-cpu-f16': '', 'llama.cpp-cpu-Q8_0': '', 'llama.cpp-cpu-Q4_K_M': '',
        'llama.cpp-vulkan-f16': '', 'llama.cpp-vulkan-Q8_0': '', 'llama.cpp-vulkan-Q4_K_M': '',
        'llama.cpp-rocm-f16': '', 'llama.cpp-rocm-Q8_0': '', 'llama.cpp-rocm-Q4_K_M': '',
    }
    bar_alphas = {
        'llama.cpp-cpu-f16': 1.0, 'llama.cpp-cpu-Q8_0': 0.7, 'llama.cpp-cpu-Q4_K_M': 0.4,
        'llama.cpp-vulkan-f16': 1.0, 'llama.cpp-vulkan-Q8_0': 0.7, 'llama.cpp-vulkan-Q4_K_M': 0.4,
        'llama.cpp-rocm-f16': 1.0, 'llama.cpp-rocm-Q8_0': 0.7, 'llama.cpp-rocm-Q4_K_M': 0.4,
        'PyTorch-iGPU-sdpa': 1.0,
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
                err = None
                fallback = False
                if bk == 'PyTorch-iGPU-sdpa':
                    entry = pt_data.get((model, 'cuda_bfloat16_sdpa', 'Linux'))
                    if entry:
                        m, e = extract_metrics_with_error(entry)
                        v = m.get(metric)
                        err = e.get(metric)
                else:
                    bk_short = bk.split('-')[1]  # 'cpu', 'vulkan', 'rocm'
                    entries = llamacpp_records.get((model, bk_short, d))
                    v, err, fallback = avg_metric(entries, metric) if entries else (None, None, False)
                key = f'{bk}-{d}'
                color_key = key if key in bar_colors else (bk if bk in bar_colors else key)
                bg.append({
                    'value': v,
                    'error': err,
                    'color': bar_colors.get(color_key, bar_colors.get(bk, '#ccc')),
                    'hatch': bar_hatches.get(key, bar_hatches.get(bk, '')),
                    'alpha': bar_alphas.get(key, bar_alphas.get(bk, 0.85)),
                    'label': key + (' *' if fallback else ''),
                })
            groups.append(bg)

        err_vals = []
        for g in groups:
            for b in g:
                if b['value'] is not None and b.get('error') is not None:
                    err_vals.append(b['error'])
                else:
                    err_vals.append(None)

        backend_fallback = {}
        for bk, d in backend_dtypes:
            if bk == 'PyTorch-iGPU-sdpa':
                continue
            bk_short = bk.split('-')[1]
            entries = llamacpp_records.get(('SmolVLM-Instruct', bk_short, d))
            if entries and any(get_effective_device(e) == 'cpu' for e in entries):
                backend_fallback[bk] = True

        color_legend = [
            Patch(facecolor='#bcbd22', label='PyTorch iGPU-SDPA'),
            Patch(facecolor='#2ca02c', label='llama.cpp-CPU' + (' *' if backend_fallback.get('llama.cpp-cpu') else '')),
            Patch(facecolor='#17becf', label='llama.cpp-Vulkan' + (' *' if backend_fallback.get('llama.cpp-vulkan') else '')),
            Patch(facecolor='#1f77b4', label='llama.cpp-ROCm' + (' *' if backend_fallback.get('llama.cpp-rocm') else '')),
        ]
        alpha_legend = [
            Patch(facecolor='black', alpha=1.0, label='F16 (α=1.0)'),
            Patch(facecolor='black', alpha=0.7, label='Q8_0 (α=0.7)'),
            Patch(facecolor='black', alpha=0.4, label='Q4_K_M (α=0.4)'),
        ]
        leg = color_legend + [Patch(facecolor='none', edgecolor='none', label='')] + alpha_legend + [Patch(facecolor='none', edgecolor='none', label='')] + [Patch(facecolor='white', edgecolor='gray', label='* = ROCm CPU fallback')]
        plot_bars(ax, groups, labels, f'{metric} ({unit})', f'SmolVLM Family: {metric}', leg, fig_w=16, fig_h=5.5, errors=err_vals if any(v is not None for v in err_vals) else None, footnote=make_footnote(metric))
        fname = f'smolvlm_family_{sanitize(metric)}.png'
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
