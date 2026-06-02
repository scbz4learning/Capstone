#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LLAMACPP_ARCHIVES="$BASE_DIR/third-party/llama-cpp"
OUT_DIR="$BASE_DIR/third-party/llama-cpp/out"
MODELS_DIR="$BASE_DIR/models"
CONFIGS_DIR="$SCRIPT_DIR/configs"
PYTHON_SCRIPT="$SCRIPT_DIR/comprehensive_profile_llamacpp.py"
VENV_PYTHON="$BASE_DIR/.venv/bin/python"
SANDBOX_DIR="/tmp/kilo/llamacpp_profiling"

mkdir -p "$SANDBOX_DIR"

# Archive -> backend name mapping
declare -A ARCHIVE_MAP
ARCHIVE_MAP["cpu"]="llama-b9357-bin-ubuntu-x64.tar.gz"
ARCHIVE_MAP["vulkan"]="llama-b9357-bin-ubuntu-vulkan-x64.tar.gz"
ARCHIVE_MAP["rocm"]="llama-b9357-bin-ubuntu-rocm-7.2-x64.tar.gz"

# Only extract archives that exist
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

echo "[Info] Available backends: ${AVAILABLE_BACKENDS[*]}"

# Detect GGUF model directories
echo "[Info] Detecting models in $MODELS_DIR"
MODEL_NAMES=()
for d in "$MODELS_DIR"/*-GGUF; do
    if [[ -d "$d" ]]; then
        name="$(basename "$d" | sed 's/-GGUF$//')"
        if ls "$d"/*.gguf >/dev/null 2>&1; then
            MODEL_NAMES+=("$name")
            echo "  Found: $name"
        fi
    fi
done

if [[ ${#MODEL_NAMES[@]} -eq 0 ]]; then
    echo "[Error] No GGUF model directories found in $MODELS_DIR"
    exit 1
fi

echo ""
echo "============================================="
echo " llama.cpp Batch Profiling"
echo "============================================="
echo " Models: ${MODEL_NAMES[*]}"
echo " Backends: ${AVAILABLE_BACKENDS[*]}"
echo " Output: $SCRIPT_DIR/profiling_logs/llamacpp"
echo "============================================="
echo ""

RUN_TS=$(date +%Y%m%d_%H%M%S)

TOTAL=$(( ${#AVAILABLE_BACKENDS[@]} * ${#MODEL_NAMES[@]} ))
CURRENT=0

for backend in "${AVAILABLE_BACKENDS[@]}"; do
    llamacpp_dir="$OUT_DIR/$backend/llama-b9357"
    if [[ ! -x "$llamacpp_dir/llama-mtmd-cli" ]]; then
        echo "[Skip] llama-mtmd-cli not found in $llamacpp_dir"
        continue
    fi

    # Detect available quants in first model to filter
    first_model="${MODEL_NAMES[0]}"
    first_dir="$MODELS_DIR/${first_model}-GGUF"
    AVAIL_QUANTS=()
    for f in "$first_dir"/*.gguf; do
        bname=$(basename "$f")
        [[ "$bname" == mmproj-* ]] && continue
        quant="${bname#"${first_model}-"}"
        quant="${quant%.gguf}"
        AVAIL_QUANTS+=("$quant")
    done

    for model_name in "${MODEL_NAMES[@]}"; do
        CURRENT=$(( CURRENT + 1 ))
        echo ""
        echo "============================================="
        echo "  [$CURRENT/$TOTAL] Backend=$backend  Model=$model_name"
        echo "============================================="

        config_file="$SANDBOX_DIR/config_${backend}_${model_name}_${RUN_TS}.json"

        # Build quant list for this specific model
        model_dir="$MODELS_DIR/${model_name}-GGUF"
        MODEL_QUANTS=()
        for f in "$model_dir"/*.gguf; do
            bname=$(basename "$f")
            [[ "$bname" == mmproj-* ]] && continue
            quant="${bname#"${model_name}-"}"
            quant="${quant%.gguf}"
            MODEL_QUANTS+=("$quant")
        done
        QUANTS_JSON=$(printf '"%s",' "${MODEL_QUANTS[@]}" | sed 's/,$//')

        cat > "$config_file" << JSONEOF
{
    "os_type": "linux",
    "task_type": "vision_autoregressive",
    "backend": "llamacpp",
    "model": "$model_name",
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
        "is_integrated": true,
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

        echo "[Run] $backend / $model_name"
        sudo "$VENV_PYTHON" "$PYTHON_SCRIPT" --config "$config_file"
        echo "[Done] $backend / $model_name completed"
    done
done

echo ""
echo "============================================="
echo " All profiling complete!"
echo " Results: $SCRIPT_DIR/profiling_logs/llamacpp/"
echo "============================================="