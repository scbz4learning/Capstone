import json
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 7

output_dir = Path('benchmark_charts_ratio')
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
    base = "* ROCm fell back to CPU execution on this device."
    if direction:
        return f"{base}\nDirection: {direction}"
    return base

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

# Color map: same semantic = same color across all charts
CFG_COLORS = {
    # PyTorch CPU variants (all envs)
    'Windows-PyTorch-CPU': '#d62728',
    'WSL-PyTorch-CPU': '#d62728',
    'Linux-PyTorch-CPU': '#d62728',
    # PyTorch iGPU-Eager variants
    'Windows-PyTorch-iGPU-Eager': '#ff7f0e',
    'WSL-PyTorch-iGPU-Eager': '#ff7f0e',
    'Linux-PyTorch-iGPU-Eager': '#ff7f0e',
    # PyTorch iGPU-SDPA variants
    'Windows-PyTorch-iGPU-SDPA': '#bcbd22',
    'WSL-PyTorch-iGPU-SDPA': '#bcbd22',
    'Linux-PyTorch-iGPU-SDPA': '#bcbd22',
    # llama.cpp CPU
    'Linux-llama.cpp-CPU': '#2ca02c',
    # llama.cpp Vulkan (real GPU)
    'Linux-llama.cpp-iGPU-Vulkan': '#17becf',
    # llama.cpp ROCm
    'Linux-llama.cpp-iGPU-ROCm': '#1f77b4',
    # VGGT configs
    'CPU-F32': '#d62728',
    'CPU-BF16': '#d62728',
    'iGPU-F32-Eager': '#ff7f0e',
    'iGPU-F32-SDPA': '#bcbd22',
    'iGPU-BF16-Eager': '#ff7f0e',
    'iGPU-BF16-SDPA': '#bcbd22',
}

def get_color_for_label(label):
    """Return rainbow color for a given bar label based on semantic meaning."""
    base = label.replace(' *', '')  # strip fallback marker
    if base in CFG_COLORS:
        return CFG_COLORS[base]
    if base.startswith('Linux-llama.cpp-'):
        parts = base[len('Linux-llama.cpp-'):].rsplit('-', 1)
        if len(parts) == 2 and parts[1] in ('f16', 'Q8_0', 'Q4_K_M'):
            base = 'Linux-llama.cpp-' + parts[0]
    if base in CFG_COLORS:
        return CFG_COLORS[base]
    if 'PyTorch' in base and 'CPU' in base:
        return '#d62728'
    if 'PyTorch' in base and 'Eager' in base:
        return '#ff7f0e'
    if 'PyTorch' in base and 'SDPA' in base:
        return '#bcbd22'
    if '\n' in base:
        kind = base.split('\n', 1)[1]
        if kind == 'PyTorch-bf16':
            return '#ff7f0e'
        if kind.startswith('cpu-'):
            return '#d62728'
        if kind.startswith('vulkan-'):
            return '#17becf'
        if kind.startswith('rocm-'):
            return '#1f77b4'
    return '#cccccc'

def get_hatch_for_label(label):
    """Return hatch pattern for a given bar label based on platform."""
    base = label.replace(' *', '')
    if 'Windows' in base:
        return 'xx'
    elif 'WSL' in base:
        return '//'
    elif 'Linux' in base:
        return ''
    return ''

# ── Load data ──
smolvlm_pt = load_json('profiling_logs/smolvlm_pytorch.json')
smolvlm_lcpp = load_json('profiling_logs/smolvlm_llamacpp.json')
vggt_data = load_json('profiling_logs/vggt_pytorch.json')

# ── Build lookup for SmolVLM-Instruct PyTorch BF16 ──
smolvlm_pt_instruct = {}
for e in smolvlm_pt:
    if e.get('_model') != 'SmolVLM-Instruct':
        continue
    key = f"{e['_environment']}-{e['config']}"
    smolvlm_pt_instruct[key] = e

# ── Build lookup for SmolVLM-Instruct llama.cpp ──
smolvlm_lcpp_instruct = {}
for e in smolvlm_lcpp:
    if e.get('model') != 'SmolVLM-Instruct' or e.get('status') != 'ok':
        continue
    cfg = e['config']
    if '_' in cfg:
        backend, quant = cfg.split('_', 1)
    else:
        backend, quant = cfg, 'unknown'
    key = f"Linux-llama.cpp-{backend}-{quant}"
    smolvlm_lcpp_instruct[key] = e

# ── Build lookup for VGGT ──
vggt_lookup = {}
for e in vggt_data:
    if e.get('status') != 'ok':
        continue
    cfg = e['config']
    env = e['_environment']
    vggt_lookup[(cfg, env)] = e

# ── Helper ──
def get_ttft(entry):
    return entry.get('latency', {}).get('ttft_ms_mean')

def get_tpot(entry):
    return entry.get('latency', {}).get('tpot_ms_mean')

def get_energy(entry):
    return entry.get('power_energy', {}).get('energy_per_inference_j')

def get_throughput(entry):
    return entry.get('latency', {}).get('batch_throughput_img_per_sec', 0)

def get_efficiency(entry):
    raw = entry.get('power_energy', {}).get('img_per_sec_watt', 0)
    return raw * 1e6  # million images/W

# ── SmolVLM-Instruct: only BF16 configs (PyTorch) ──
PT_CFG_DISPLAY = {
    'Windows-cpu_bfloat16_none': 'Windows-PyTorch-CPU',
    'WSL-cpu_bfloat16_none': 'WSL-PyTorch-CPU',
    'Linux-cpu_bfloat16_none': 'Linux-PyTorch-CPU',
    'Windows-cuda_bfloat16_eager': 'Windows-PyTorch-iGPU-Eager',
    'WSL-cuda_bfloat16_eager': 'WSL-PyTorch-iGPU-Eager',
    'Linux-cuda_bfloat16_eager': 'Linux-PyTorch-iGPU-Eager',
    'Windows-cuda_bfloat16_sdpa': 'Windows-PyTorch-iGPU-SDPA',
    'WSL-cuda_bfloat16_sdpa': 'WSL-PyTorch-iGPU-SDPA',
    'Linux-cuda_bfloat16_sdpa': 'Linux-PyTorch-iGPU-SDPA',
}

SMOLVLM_BF16_CFGS = []
for key, entry in smolvlm_pt_instruct.items():
    display = PT_CFG_DISPLAY.get(key, key)
    SMOLVLM_BF16_CFGS.append({
        'label': display,
        'ttft': get_ttft(entry),
        'tpot': get_tpot(entry),
        'energy': get_energy(entry),
    })

LCPP_CFG_DISPLAY = {
    'Linux-llama.cpp-cpu-f16': 'Linux-llama.cpp-CPU-f16',
    'Linux-llama.cpp-cpu-Q8_0': 'Linux-llama.cpp-CPU-Q8_0',
    'Linux-llama.cpp-cpu-Q4_K_M': 'Linux-llama.cpp-CPU-Q4_K_M',
    'Linux-llama.cpp-vulkan-f16': 'Linux-llama.cpp-iGPU-Vulkan-f16',
    'Linux-llama.cpp-vulkan-Q8_0': 'Linux-llama.cpp-iGPU-Vulkan-Q8_0',
    'Linux-llama.cpp-vulkan-Q4_K_M': 'Linux-llama.cpp-iGPU-Vulkan-Q4_K_M',
    'Linux-llama.cpp-rocm-f16': 'Linux-llama.cpp-iGPU-ROCm-f16',
    'Linux-llama.cpp-rocm-Q8_0': 'Linux-llama.cpp-iGPU-ROCm-Q8_0',
    'Linux-llama.cpp-rocm-Q4_K_M': 'Linux-llama.cpp-iGPU-ROCm-Q4_K_M',
}

# ── Collect ALL SmolVLM-Instruct configs (BF16 + llama.cpp) ──
SMOLVLM_ALL_CFGS = []
for key, entry in smolvlm_pt_instruct.items():
    display = PT_CFG_DISPLAY.get(key, key)
    SMOLVLM_ALL_CFGS.append({
        'label': display,
        'ttft': get_ttft(entry),
        'tpot': get_tpot(entry),
        'energy': get_energy(entry),
    })
for key, entry in smolvlm_lcpp_instruct.items():
    display = LCPP_CFG_DISPLAY.get(key, key)
    is_fallback = get_effective_device(entry) == 'cpu'
    SMOLVLM_ALL_CFGS.append({
        'label': display + (' *' if is_fallback else ''),
        'ttft': get_ttft(entry),
        'tpot': get_tpot(entry),
        'energy': get_energy(entry),
        'is_fallback': is_fallback,
    })

# Best f16 values (used as 1.0x baseline)
best_ttft = min(c['ttft'] for c in SMOLVLM_ALL_CFGS
                if c['ttft'] is not None and 'f16' in c['label'])
best_tpot = min(c['tpot'] for c in SMOLVLM_ALL_CFGS
                if c['tpot'] is not None and 'f16' in c['label'])
best_energy = min(c['energy'] for c in SMOLVLM_ALL_CFGS
                  if c['energy'] is not None and 'f16' in c['label'])

print(f'SmolVLM f16 baseline: TTFT={best_ttft:.0f}ms TPOT={best_tpot:.1f}ms Energy={best_energy:.0f}J')

# ── VGGT: all configs (F32 + BF16) ──
VGGT_ALL_CFGS = ['cpu_float32_none', 'cpu_bfloat16_none',
                 'cuda_float32_eager', 'cuda_float32_sdpa',
                 'cuda_bfloat16_eager', 'cuda_bfloat16_sdpa']
VGGT_ENVS = ['Windows', 'WSL', 'Linux']

VGGT_CFG_DISPLAY = {
    'cuda_float32_eager': 'iGPU-F32-Eager',
    'cuda_float32_sdpa': 'iGPU-F32-SDPA',
    'cuda_bfloat16_eager': 'iGPU-BF16-Eager',
    'cuda_bfloat16_sdpa': 'iGPU-BF16-SDPA',
    'cpu_float32_none': 'CPU-F32',
    'cpu_bfloat16_none': 'CPU-BF16',
}

VGGT_ENV_COLORS = {
    'Windows': '#1f77b4', 'WSL': '#ff7f0e', 'Linux': '#2ca02c',
}

VGGT_ENV_HATCH = {'Windows': '', 'WSL': '//', 'Linux': 'xx'}

# Best BF16 VGGT values
vggt_bf16_entries = []
for cfg in ['cpu_bfloat16_none', 'cuda_bfloat16_eager', 'cuda_bfloat16_sdpa']:
    for env in VGGT_ENVS:
        e = vggt_lookup.get((cfg, env))
        if e:
            vggt_bf16_entries.append(e)

best_vggt_throughput = max(get_throughput(e) for e in vggt_bf16_entries)
best_vggt_efficiency = max(get_efficiency(e) for e in vggt_bf16_entries)
print(f'VGGT BF16 baseline: Throughput={best_vggt_throughput:.4f} img/s Efficiency={best_vggt_efficiency:.2f} million/W')

# ── Chart helpers ──
def make_bar_chart(configs, metric_key, ylabel, title, fname, footnote=None):
    valid = [(c, c[metric_key]) for c in configs if c[metric_key] is not None]
    valid.sort(key=lambda x: x[1])
    labels = [c['label'] for c, _ in valid]
    values = [v for _, v in valid]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [get_color_for_label(c['label']) for c in configs]
    alphas = []
    for c in configs:
        lbl = c['label']
        if 'BF16' in lbl or 'f16' in lbl:
            alphas.append(1.0)
        elif 'Q8_0' in lbl:
            alphas.append(0.7)
        elif 'Q4_K_M' in lbl:
            alphas.append(0.4)
        else:
            alphas.append(0.85)
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.4)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    hatches = [get_hatch_for_label(c['label']) for c in configs]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for bar, val in zip(bars, values):
        w = bar.get_width()
        ax.text(w, bar.get_y() + bar.get_height()/2, f'{w:.1f}x', ha='left', va='center', fontsize=6)

    ax.axvline(x=1, color='green', linestyle='--', alpha=0.6, linewidth=1)
    fig.tight_layout()

    pos = ax.get_position()
    ax_r = pos.x0 + pos.width
    ax_t = pos.y0 + pos.height
    ax_b = pos.y0

    color_legend = [
        Patch(facecolor='#d62728', label='PyTorch-CPU'),
        Patch(facecolor='#ff7f0e', label='PyTorch-iGPU-Eager'),
        Patch(facecolor='#bcbd22', label='PyTorch-iGPU-SDPA'),
        Patch(facecolor='#2ca02c', label='llama.cpp-CPU'),
        Patch(facecolor='#17becf', label='llama.cpp-Vulkan'),
        Patch(facecolor='#1f77b4', label='llama.cpp-ROCm'),
    ]
    hatch_legend = [
        Patch(facecolor='white', edgecolor='gray', hatch='xx', label='Windows'),
        Patch(facecolor='white', edgecolor='gray', hatch='//', label='WSL'),
        Patch(facecolor='white', edgecolor='gray', hatch='', label='Linux'),
    ]
    alpha_legend = [
        Patch(facecolor='black', alpha=1.0, label='F16 (α=1.0)'),
        Patch(facecolor='black', alpha=0.7, label='Q8_0 (α=0.7)'),
        Patch(facecolor='black', alpha=0.4, label='Q4_K_M (α=0.4)'),
    ]
    leg = color_legend + [Patch(facecolor='none', edgecolor='none', label='')] + hatch_legend + [Patch(facecolor='none', edgecolor='none', label='')] + alpha_legend
    ax.legend(
        handles=leg,
        loc="upper left",
        bbox_to_anchor=(ax_r + 0.02, ax_t),
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
            fontsize=8,
            bbox=dict(facecolor='#f5f5f5', edgecolor='#cccccc', boxstyle='round,pad=0.3'),
        )

    out_path = output_dir / fname
    print(f'  Saving to: {out_path.absolute()}')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Generated: {fname}')

def make_vggt_bar_chart(configs, envs, metric_extractor, ylabel, title, fname, best_val, footnote=None):
    entries = []
    for cfg in configs:
        for env in envs:
            e = vggt_lookup.get((cfg, env))
            if e is None:
                continue
            val = metric_extractor(e)
            if val is not None and val > 0:
                display_cfg = VGGT_CFG_DISPLAY.get(cfg, cfg)
                label = f'{env}-{display_cfg}'
                entries.append({
                    'label': label,
                    'mult': val / best_val,
                    'color': get_color_for_label(display_cfg),
                    'hatch': VGGT_ENV_HATCH.get(env, ''),
                })

    if not entries:
        print(f'  No data for {fname}, skipping.')
        print(f'  Debug: lookup size={len(vggt_lookup)}, configs={VGGT_ALL_CFGS}, envs={VGGT_ENVS}')
        for cfg in VGGT_ALL_CFGS:
            for env in VGGT_ENVS:
                e = vggt_lookup.get((cfg, env))
                print(f'    {cfg} / {env}: {e is not None}')
        return

    entries.sort(key=lambda x: x['mult'])

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = [e['color'] for e in entries]
    hatches = [e.get('hatch', '') for e in entries]
    lws = [1.5 if 'F32' in e['label'] else 0.5 for e in entries]
    bars = ax.barh(range(len(entries)), [e['mult'] for e in entries],
                   color=colors, edgecolor='black', linewidth=0.4)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for bar, lw in zip(bars, lws):
        bar.set_linewidth(lw)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e['label'] for e in entries], fontsize=6)
    ax.set_xlabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for bar, entry in zip(bars, entries):
        w = bar.get_width()
        ax.text(w, bar.get_y() + bar.get_height()/2, f'{w:.2f}x', ha='left', va='center', fontsize=5)

    ax.axvline(x=1, color='green', linestyle='--', alpha=0.6, linewidth=1)
    fig.tight_layout()

    pos = ax.get_position()
    ax_r = pos.x0 + pos.width
    ax_t = pos.y0 + pos.height
    ax_b = pos.y0

    color_legend = [
        Patch(facecolor=get_color_for_label('CPU-F32'), label='CPU'),
        Patch(facecolor=get_color_for_label('iGPU-F32-Eager'), label='iGPU-Eager'),
        Patch(facecolor=get_color_for_label('iGPU-F32-SDPA'), label='iGPU-SDPA'),
    ]
    hatch_legend = [
        Patch(facecolor='white', edgecolor='gray', hatch='xx', label='Windows'),
        Patch(facecolor='white', edgecolor='gray', hatch='//', label='WSL'),
        Patch(facecolor='white', edgecolor='gray', hatch='', label='Linux'),
    ]
    lw_legend = [
        Patch(facecolor='white', edgecolor='black', linewidth=2, label='F32'),
        Patch(facecolor='white', edgecolor='black', linewidth=0.5, label='BF16'),
    ]
    leg = color_legend + [Patch(facecolor='none', edgecolor='none', label='')] + hatch_legend + [Patch(facecolor='none', edgecolor='none', label='')] + lw_legend
    ax.legend(
        handles=leg,
        loc="upper left",
        bbox_to_anchor=(ax_r + 0.02, ax_t),
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
            fontsize=8,
            bbox=dict(facecolor='#f5f5f5', edgecolor='#cccccc', boxstyle='round,pad=0.3'),
        )

    fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Generated: {fname}')

# ── SmolVLM-Instruct BF16 Charts ──
print('\nSmolVLM-Instruct BF16 charts:')

TTFT_DATA = [{'label': c['label'], 'ttft_mult': c['ttft'] / best_ttft}
             for c in SMOLVLM_ALL_CFGS if c['ttft']]
TTFT_DATA.sort(key=lambda x: x['ttft_mult'])
make_bar_chart(TTFT_DATA, 'ttft_mult', 'Multiplier (relative to best BF16)',
               f'SmolVLM-Instruct: TTFT (BF16 best={best_ttft:.0f}ms)',
               'smolvlm_instruct_ttft_ratio.png', footnote=make_footnote('TTFT (ms)'))

TPOT_DATA = [{'label': c['label'], 'tpot_mult': c['tpot'] / best_tpot}
             for c in SMOLVLM_ALL_CFGS if c['tpot']]
TPOT_DATA.sort(key=lambda x: x['tpot_mult'])
make_bar_chart(TPOT_DATA, 'tpot_mult', 'Multiplier (relative to best BF16)',
               f'SmolVLM-Instruct: TPOT (BF16 best={best_tpot:.1f}ms)',
                'smolvlm_instruct_tpot_ratio.png', footnote=make_footnote('TPOT (ms)'))

ENERGY_DATA = [{'label': c['label'], 'energy_mult': c['energy'] / best_energy}
               for c in SMOLVLM_ALL_CFGS if c['energy']]
ENERGY_DATA.sort(key=lambda x: x['energy_mult'])
make_bar_chart(ENERGY_DATA, 'energy_mult', 'Multiplier (relative to best BF16)',
               f'SmolVLM-Instruct: Energy per Inference (BF16 best={best_energy:.0f}J)',
                'smolvlm_instruct_energy_ratio.png', footnote=make_footnote('Energy per Inference (J)'))

# ── VGGT Charts (all configs) ──
print('\nVGGT charts:')

make_vggt_bar_chart(VGGT_ALL_CFGS, VGGT_ENVS,
                    get_throughput, 'Multiplier (relative to best BF16)',
                    f'VGGT: Throughput (BF16 best={best_vggt_throughput:.4f} img/s)',
                    'vggt_throughput_ratio.png', best_vggt_throughput)

make_vggt_bar_chart(VGGT_ALL_CFGS, VGGT_ENVS,
                    get_efficiency, 'Multiplier (relative to best BF16)',
                    f'VGGT: Efficiency (BF16 best={best_vggt_efficiency:.2f} million/W)',
                    'vggt_efficiency_ratio.png', best_vggt_efficiency)

# ── SmolVLM Family: ALL configs across all models, ratio vs baseline ──
print('\nSmolVLM family ALL configs ratio charts:')

ALL_MODELS = ['SmolVLM-256M-Instruct', 'SmolVLM-500M-Instruct', 'SmolVLM-Instruct',
              'SmolVLM2-256M-Video-Instruct', 'SmolVLM2-500M-Video-Instruct', 'SmolVLM2-2.2B-Instruct']

DTYPE_COLORS = {
    'Q4_K_M': '#d62728',
    'Q8_0': '#ff7f0e',
    'f16': '#bcbd22',
    'bf16': '#2ca02c',
}

BASELINE_TTFT = 115.57379999999998   # SmolVLM-Instruct | llama.cpp-vulkan | f16
BASELINE_TPUT = 46.749975390624996
BASELINE_ENERGY = 338.2762097799765

# Build all config entries (PyTorch + llama.cpp)
all_family_entries = []

# PyTorch BF16
for e in smolvlm_pt:
    if e.get('status') != 'ok': continue
    model = e.get('_model', '')
    if model not in ALL_MODELS: continue
    if e.get('config') != 'cuda_bfloat16_sdpa': continue
    lat = e.get('latency', {})
    pe = e.get('power_energy', {})
    mem = e.get('memory', {})
    mem_gb = mem.get('cpu_rss_mb', 0) / 1024 if 'cpu_rss_mb' in mem else mem.get('peak_mem_allocated_gb', mem.get('vram_gb', 0))
    tj = pe.get('tokens_per_joule_p50')
    all_family_entries.append({
        'label': f'{model}\nPyTorch-bf16',
        'ttft': lat.get('ttft_ms_mean'),
        'tpot': lat.get('tpot_ms_mean'),
        'energy': pe.get('energy_per_inference_j'),
        'dtype': 'bf16',
    })

# llama.cpp
for e in smolvlm_lcpp:
    if e.get('status') != 'ok': continue
    model = e.get('model', '')
    if model not in ALL_MODELS: continue
    cfg = e['config']
    if '_' in cfg:
        backend, quant = cfg.split('_', 1)
    else:
        backend, quant = cfg, 'unknown'
    lat = e.get('latency', {})
    pe = e.get('power_energy', {})
    mem = e.get('memory', {})
    peak_gb = mem.get('peak_mem_allocated_gb', 0)
    if peak_gb == 0:
        peak_gb = mem.get('cpu_rss_mb', 0) / 1024
    is_fallback = get_effective_device(e) == 'cpu'
    all_family_entries.append({
        'label': f"{model}\n{backend}-{quant}{' *' if is_fallback else ''}",
        'ttft': lat.get('ttft_ms_mean'),
        'tpot': lat.get('tpot_ms_mean'),
        'energy': pe.get('energy_per_inference_j'),
        'dtype': quant,
        'is_fallback': is_fallback,
    })

def make_family_ratio_chart(metric, ylabel, title, fname, footnote=None):
    # Map short metric keys to display names used in METRIC_DIRECTION
    metric_display_map = {
        'ttft': 'TTFT (ms)',
        'tpot': 'TPOT (ms)',
        'energy': 'Energy per Inference (J)',
    }
    metric_key = metric_display_map.get(metric, metric)
    if footnote is None:
        footnote = make_footnote(metric_key)
    valid = [e for e in all_family_entries if e[metric] is not None]
    baseline_map = {'ttft': BASELINE_TTFT, 'tpot': BASELINE_TPUT, 'energy': BASELINE_ENERGY}
    baseline = baseline_map[metric]
    for e in valid:
        e['mult'] = e[metric] / baseline
    valid.sort(key=lambda x: x['mult'])

    fig, ax = plt.subplots(figsize=(14, 12))
    colors = [get_color_for_label(e['label']) for e in valid]
    hatches = [get_hatch_for_label(e['label']) for e in valid]
    alphas = []
    for e in valid:
        dtype = e.get('dtype', '')
        if dtype in ('bf16', 'f16'):
            alphas.append(1.0)
        elif dtype == 'Q8_0':
            alphas.append(0.7)
        elif dtype == 'Q4_K_M':
            alphas.append(0.4)
        else:
            alphas.append(0.85)
    bars = ax.barh(range(len(valid)), [e['mult'] for e in valid],
                   color=colors, edgecolor='black', linewidth=0.4)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels([e['label'] for e in valid], fontsize=6)
    ax.set_xlabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for bar, entry in zip(bars, valid):
        w = bar.get_width()
        ax.text(w, bar.get_y() + bar.get_height()/2, f'{w:.2f}x', ha='left', va='center', fontsize=5)

    ax.axvline(x=1, color='green', linestyle='--', alpha=0.6, linewidth=1)

    fig.tight_layout()

    pos = ax.get_position()
    ax_r = pos.x0 + pos.width
    ax_t = pos.y0 + pos.height
    ax_b = pos.y0

    color_legend = [
        Patch(facecolor='#d62728', label='PyTorch-CPU'),
        Patch(facecolor='#ff7f0e', label='PyTorch-iGPU-Eager'),
        Patch(facecolor='#bcbd22', label='PyTorch-iGPU-SDPA'),
        Patch(facecolor='#2ca02c', label='llama.cpp-CPU'),
        Patch(facecolor='#17becf', label='llama.cpp-Vulkan'),
        Patch(facecolor='#1f77b4', label='llama.cpp-ROCm'),
    ]
    hatch_legend = [
        Patch(facecolor='white', edgecolor='gray', hatch='xx', label='Windows'),
        Patch(facecolor='white', edgecolor='gray', hatch='//', label='WSL'),
        Patch(facecolor='white', edgecolor='gray', hatch='', label='Linux'),
    ]
    alpha_legend = [
        Patch(facecolor='black', alpha=1.0, label='F16 (α=1.0)'),
        Patch(facecolor='black', alpha=0.7, label='Q8_0 (α=0.7)'),
        Patch(facecolor='black', alpha=0.4, label='Q4_K_M (α=0.4)'),
    ]
    leg = color_legend + [Patch(facecolor='none', edgecolor='none', label='')] + hatch_legend + [Patch(facecolor='none', edgecolor='none', label='')] + alpha_legend
    ax.legend(
        handles=leg,
        loc="upper left",
        bbox_to_anchor=(ax_r + 0.02, ax_t),
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
            fontsize=8,
            bbox=dict(facecolor='#f5f5f5', edgecolor='#cccccc', boxstyle='round,pad=0.3'),
        )

    fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Generated: {fname}')

make_family_ratio_chart('ttft', 'Multiplier (relative to SmolVLM-Instruct BF16)',
                        'SmolVLM Family: ALL configs — TTFT',
                        'smolvlm_family_dtype_ttft_ratio.png', footnote=None)

make_family_ratio_chart('tpot', 'Multiplier (relative to SmolVLM-Instruct BF16)',
                        'SmolVLM Family: ALL configs — TPOT',
                        'smolvlm_family_dtype_tpot_ratio.png', footnote=None)

make_family_ratio_chart('energy', 'Multiplier (relative to SmolVLM-Instruct BF16)',
                        'SmolVLM Family: ALL configs — Energy',
                        'smolvlm_family_dtype_energy_ratio.png', footnote=None)

print(f'\nDone → {output_dir.absolute()}')
