#!/bin/bash
# Download SmolLM/SmolLM2 135M models.
# Usage: ./scripts/download.sh [model_key]
#   model_key: all (default), npu, hybrid, smollm-npu, smollm2-npu, smollm-hybrid, smollm2-hybrid

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

ALL_KEYS="smollm-npu smollm2-npu smollm-hybrid smollm2-hybrid"

SELECTION="${1:-all}"
case "$SELECTION" in
    all)    KEYS="$ALL_KEYS" ;;
    npu)    KEYS="smollm-npu smollm2-npu" ;;
    hybrid) KEYS="smollm-hybrid smollm2-hybrid" ;;
    *)      KEYS="$SELECTION" ;;
esac

if ! git lfs version &>/dev/null; then
    echo "ERROR: git-lfs not installed."
    echo "Install from: https://github.com/git-lfs/git-lfs/releases"
    exit 1
fi

mkdir -p "$MODELS_DIR"

download_model() {
    local key="$1"
    local repo="$2"
    local dir="$3"

    if [ -d "$dir" ] && [ -f "$dir/genai_config.json" ]; then
        local onnx_file
        onnx_file=$(find "$dir" -maxdepth 1 -name "*.onnx" 2>/dev/null | head -1)
        if [ -n "$onnx_file" ] && file "$onnx_file" | grep -qE "ONNX|data|Protobuf"; then
            echo "[SKIP] $key: already downloaded at $dir"
            return 0
        fi
        echo "[UPDATE] $key: re-downloading (incomplete)"
        rm -rf "$dir"
    fi

    echo "[DOWNLOAD] $key: https://huggingface.co/$repo"
    git clone "https://huggingface.co/$repo" "$dir" 2>&1 | tail -3
    echo "[OK] $key -> $dir"
}

for key in $KEYS; do
    case "$key" in
        smollm-npu)     download_model "$key" "amd/SmolLM-135M-Instruct_rai_1.7.1_npu_4K"     "$MODELS_DIR/SmolLM-135M-Instruct_rai_1.7.1_npu_4K" ;;
        smollm2-npu)    download_model "$key" "amd/SmolLM2-135M-Instruct_rai_1.7.1_npu_4K"    "$MODELS_DIR/SmolLM2-135M-Instruct_rai_1.7.1_npu_4K" ;;
        smollm-hybrid)  download_model "$key" "amd/SmolLM-135M-Instruct_rai_1.7.1_hybrid"    "$MODELS_DIR/SmolLM-135M-Instruct_rai_1.7.1_hybrid" ;;
        smollm2-hybrid) download_model "$key" "amd/SmolLM2-135M-Instruct_rai_1.7.1_hybrid"   "$MODELS_DIR/SmolLM2-135M-Instruct_rai_1.7.1_hybrid" ;;
        *) echo "[WARN] Unknown model key: $key (valid: $ALL_KEYS)" ;;
    esac
done

echo "[download] All done."