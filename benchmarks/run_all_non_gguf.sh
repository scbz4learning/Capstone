#!/bin/bash
set -e

MODELS_ROOT="../models"
SCRIPTS_DIR="../benchmarks"
CONFIGS_DIR="${SCRIPTS_DIR}/configs"
TEMP_CONFIG_DIR="/tmp/bench_configs_$$"

mkdir -p "$TEMP_CONFIG_DIR"
trap "rm -rf $TEMP_CONFIG_DIR" EXIT

SYS_PYTHON="${SYS_PYTHON:-python3}"

# Check passwordless sudo upfront
sudo -n true 2>/dev/null || {
    echo "[Error] Passwordless sudo is required for background execution."
    exit 1
}

smolvlm_models=()
vggt_models=()

for model_dir in "$MODELS_ROOT"/*/; do
    model_name=$(basename "$model_dir")
    if [[ "$model_name" == *-GGUF ]]; then
        continue
    fi
    case "$model_name" in
        VGGT-*)
            vggt_models+=("$model_name")
            ;;
        SmolVLM*)
            smolvlm_models+=("$model_name")
            ;;
        *)
            echo "[Warning] Unknown model type: $model_name, skipping"
            ;;
    esac
done

run_smolvlm() {
    local model_name="$1"
    local model_path="${MODELS_ROOT}/${model_name}"
    local config_file="${TEMP_CONFIG_DIR}/smolvlm_${model_name}.json"

    jq --arg model "$model_path" \
       '.model = $model | .execution.device = ["cuda"]' \
        "${CONFIGS_DIR}/smolvlm-linux.json" > "$config_file"

    echo ""
    echo "=============================================="
    echo "  [SmolVLM] Starting: $model_name"
    echo "=============================================="
    sudo -n -E "$SYS_PYTHON" "${SCRIPTS_DIR}/comprehensive_profile_smolvlm.py" \
        --config "$config_file"
    echo "[SmolVLM] Completed: $model_name"
}

run_vggt() {
    local model_name="$1"
    local model_path="${MODELS_ROOT}/${model_name}"
    local config_file="${TEMP_CONFIG_DIR}/vggt_${model_name}.json"

    jq --arg model "$model_path" \
       '.model = $model | .execution.device = ["cuda"]' \
        "${CONFIGS_DIR}/vggt-linux.json" > "$config_file"

    echo ""
    echo "=============================================="
    echo "  [VGGT] Starting: $model_name"
    echo "=============================================="
    sudo -n -E "$SYS_PYTHON" "${SCRIPTS_DIR}/comprehensive_profile_vggt.py" \
        --config "$config_file"
    echo "[VGGT] Completed: $model_name"
}

echo "=== Non-GGUF Models Found ==="
echo "SmolVLM: ${smolvlm_models[*]}"
echo "VGGT:    ${vggt_models[*]}"
echo ""

for model in "${smolvlm_models[@]}"; do
    run_smolvlm "$model"
done

for model in "${vggt_models[@]}"; do
    run_vggt "$model"
done

echo ""
echo "=============================================="
echo "  All benchmarks completed!"
echo "=============================================="