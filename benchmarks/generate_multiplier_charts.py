import json
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 7

output_dir = Path('benchmark_charts_ratio')
output_dir.mkdir(exist_ok=True)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

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
    SMOLVLM_ALL_CFGS.append({
        'label': display,
        'ttft': get_ttft(entry),
        'tpot': get_tpot(entry),
        'energy': get_energy(entry),
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
def make_bar_chart(configs, metric_key, ylabel, title, fname):
    valid = [(c, c[metric_key]) for c in configs if c[metric_key] is not None]
    valid.sort(key=lambda x: x[1])
    labels = [c['label'] for c, _ in valid]
    values = [v for _, v in valid]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#d62728' if v > 10 else '#ff7f0e' if v > 5 else '#1f77b4' for v in values]
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.4)
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
    fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Generated: {fname}')

def make_vggt_bar_chart(configs, envs, metric_extractor, ylabel, title, fname, best_val):
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
                    'color': VGGT_ENV_COLORS[env],
                })

    if not entries:
        print(f'  No data for {fname}, skipping.')
        return

    entries.sort(key=lambda x: x['mult'])

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(range(len(entries)), [e['mult'] for e in entries],
                   color=[e['color'] for e in entries],
                   edgecolor='black', linewidth=0.4)
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
               'smolvlm_instruct_ttft_ratio.png')

TPOT_DATA = [{'label': c['label'], 'tpot_mult': c['tpot'] / best_tpot}
             for c in SMOLVLM_ALL_CFGS if c['tpot']]
TPOT_DATA.sort(key=lambda x: x['tpot_mult'])
make_bar_chart(TPOT_DATA, 'tpot_mult', 'Multiplier (relative to best BF16)',
               f'SmolVLM-Instruct: TPOT (BF16 best={best_tpot:.1f}ms)',
               'smolvlm_instruct_tpot_ratio.png')

ENERGY_DATA = [{'label': c['label'], 'energy_mult': c['energy'] / best_energy}
               for c in SMOLVLM_ALL_CFGS if c['energy']]
ENERGY_DATA.sort(key=lambda x: x['energy_mult'])
make_bar_chart(ENERGY_DATA, 'energy_mult', 'Multiplier (relative to best BF16)',
               f'SmolVLM-Instruct: Energy per Inference (BF16 best={best_energy:.0f}J)',
               'smolvlm_instruct_energy_ratio.png')

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
    'bf16': '#1f77b4',
    'f16': '#ff7f0e',
    'Q8_0': '#2ca02c',
    'Q4_K_M': '#d62728',
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
    mem_gb = mem.get('cpu_rss_mb', 0) / 1024 if 'cpu_rss_mb' in mem else mem.get('vram_gb', mem.get('peak_mem_allocated_gb', 0))
    tj = pe.get('tokens_per_joule_p50')
    all_family_entries.append({
        'label': f'{model}\n{backend}-{quant}',
        'ttft': lat.get('ttft_ms_mean'),
        'tpot': lat.get('tpot_ms_mean'),
        'energy': pe.get('energy_per_inference_j'),
        'dtype': quant,
    })

def make_family_ratio_chart(metric, ylabel, title, fname):
    valid = [e for e in all_family_entries if e[metric] is not None]
    baseline_map = {'ttft': BASELINE_TTFT, 'tpot': BASELINE_TPUT, 'energy': BASELINE_ENERGY}
    baseline = baseline_map[metric]
    for e in valid:
        e['mult'] = e[metric] / baseline
    valid.sort(key=lambda x: x['mult'])

    fig, ax = plt.subplots(figsize=(14, 12))
    colors = [DTYPE_COLORS.get(e['dtype'], '#ccc') for e in valid]
    bars = ax.barh(range(len(valid)), [e['mult'] for e in valid],
                   color=colors, edgecolor='black', linewidth=0.4)
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels([e['label'] for e in valid], fontsize=6)
    ax.set_xlabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for bar, entry in zip(bars, valid):
        w = bar.get_width()
        ax.text(w, bar.get_y() + bar.get_height()/2, f'{w:.2f}x', ha='left', va='center', fontsize=5)

    ax.axvline(x=1, color='green', linestyle='--', alpha=0.6, linewidth=1)

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=DTYPE_COLORS[dt], label=dt) for dt in ['bf16', 'f16', 'Q8_0', 'Q4_K_M']]
    fig.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.0, 1), fontsize=7, ncol=1)

    fig.tight_layout()
    fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Generated: {fname}')

make_family_ratio_chart('ttft', 'Multiplier (relative to SmolVLM-Instruct BF16)',
                        'SmolVLM Family: ALL configs — TTFT',
                        'smolvlm_family_dtype_ttft_ratio.png')

make_family_ratio_chart('tpot', 'Multiplier (relative to SmolVLM-Instruct BF16)',
                        'SmolVLM Family: ALL configs — TPOT',
                        'smolvlm_family_dtype_tpot_ratio.png')

make_family_ratio_chart('energy', 'Multiplier (relative to SmolVLM-Instruct BF16)',
                        'SmolVLM Family: ALL configs — Energy',
                        'smolvlm_family_dtype_energy_ratio.png')

print(f'\nDone → {output_dir.absolute()}')
