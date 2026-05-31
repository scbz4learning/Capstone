#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="/home/bokai/capstone"
LLAMACPP_ARCHIVES="$BASE_DIR/third-party/llama-cpp"
OUT_DIR="$BASE_DIR/third-party/llama-cpp/out"
MODELS_DIR="$BASE_DIR/models"
PYTHON_SCRIPT="$BASE_DIR/benchmarks/comprehensive_profile_llamacpp.py"
VENV_PYTHON="$BASE_DIR/.venv/bin/python"
SANDBOX_DIR="/tmp/kilo/llamacpp_profiling"

mkdir -p "$SANDBOX_DIR"

declare -A ARCHIVE_MAP
ARCHIVE_MAP["cpu"]="llama-b9357-bin-ubuntu-x64.tar.gz"
ARCHIVE_MAP["vulkan"]="llama-b9357-bin-ubuntu-vulkan-x64.tar.gz"
ARCHIVE_MAP["rocm"]="llama-b9357-bin-ubuntu-rocm-7.2-x64.tar.gz"

AVAILABLE_BACKENDS=()
for backend in cpu vulkan rocm; do
    archive="$LLAMACPP_ARCHIVES/${ARCHIVE_MAP[$backend]}"
    if [[ -f "$archive" ]]; then
        AVAILABLE_BACKENDS+=("$backend")
        extract_dir="$OUT_DIR/$backend"
        if [[ ! -d "$extract_dir/llama-b9357" ]]; then
            echo "[Setup] Extracting $backend backend..."
            mkdir -p "$extract_dir"
            tar xzf "$archive" -C "$extract_dir"
        else
            echo "[Setup] $backend already extracted at $extract_dir/llama-b9357"
        fi
    fi
done

if [[ ${#AVAILABLE_BACKENDS[@]} -eq 0 ]]; then
    echo "[Error] No llama.cpp archives found in $LLAMACPP_ARCHIVES"
    echo "  Expected one of: ${ARCHIVE_MAP[*]}"
    exit 1
fi

model_dir="$MODELS_DIR/SmolVLM-Instruct-GGUF"
if [[ ! -d "$model_dir" ]]; then
    echo "[Error] Model directory not found: $model_dir"
    exit 1
fi

echo ""
echo "============================================="
echo " llama.cpp Batch Profiling - SmolVLM-Instruct"
echo "============================================="
echo " Backends: ${AVAILABLE_BACKENDS[*]}"
echo " Output: $SCRIPT_DIR/profiling_logs/llamacpp"
echo "============================================="
echo ""

RUN_TS=$(date +%Y%m%d_%H%M%S)
MODEL_QUANTS=()
for f in "$model_dir"/*.gguf; do
    bname=$(basename "$f")
    [[ "$bname" == mmproj-* ]] && continue
    quant="${bname#"SmolVLM-Instruct-"}"
    quant="${quant%.gguf}"
    MODEL_QUANTS+=("$quant")
done
QUANTS_JSON=$(printf '"%s",' "${MODEL_QUANTS[@]}" | sed 's/,$//')

for backend in "${AVAILABLE_BACKENDS[@]}"; do
    llamacpp_dir="$OUT_DIR/$backend/llama-b9357"
    if [[ ! -x "$llamacpp_dir/llama-mtmd-cli" ]]; then
        echo "[Skip] llama-mtmd-cli not found in $llamacpp_dir"
        continue
    fi

    echo ""
    echo "============================================="
    echo "  Backend=$backend  Model=SmolVLM-Instruct"
    echo "============================================="

    config_file="$SANDBOX_DIR/config_${backend}_SmolVLM-Instruct_${RUN_TS}.json"
    cat > "$config_file" << JSONEOF
{
    "os_type": "linux",
    "task_type": "vision_autoregressive",
    "backend": "llamacpp",
    "model": "SmolVLM-Instruct",
    "output_dir": "$SCRIPT_DIR/profiling_logs/llamacpp",
    "models_root": "$MODELS_DIR",
    "llamacpp": {
        "dir": "$llamacpp_dir",
        "ngl": $(if [[ "$backend" == "cpu" ]]; then echo "null"; else echo "99"; fi),
        "threads": null,
        "flash_attn": "auto"
    },
    "execution": {
        "device": "$backend",
        "quantizations": [$QUANTS_JSON],
        "is_integrated": $(if [[ "$backend" == "cpu" ]]; then echo "true"; else echo "false"; fi),
        "sampling_randomness": false,
        "temperature": null,
        "passes": {
            "warmup": { "num_warmup": 10 },
            "end_to_end": { "num_test": 20 },
            "power": { "num_test": 10 }
        }
    },
    "inputs": {
        "prompt": "Describe the images briefly.",
        "prompt_size": 64,
        "images": [],
        "image_size": 384,
        "num_images": 2
    },
    "output": {
        "output_tokens": 128
    },
    "target_metrics": {
        "warmup": ["warmup_latency"],
        "end_to_end": ["TTFT", "TPOT", "Total_latency", "tokens/sec", "peak_mem"],
        "power": ["avg_power", "energy_per_inference", "fps_watt"],
        "bandwidth_analysis": false
    }
}
JSONEOF

    echo "[Run] $backend / SmolVLM-Instruct"
    sudo "$VENV_PYTHON" "$PYTHON_SCRIPT" --config "$config_file"
    echo "[Done] $backend / SmolVLM-Instruct completed"
done

echo ""
echo "============================================="
echo " Profiling complete!"
echo " Results: $SCRIPT_DIR/profiling_logs/llamacpp/"
echo "============================================="
