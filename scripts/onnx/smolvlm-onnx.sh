#!/bin/bash
# Download SmolVLM-Instruct at the old ONNX-compatible commit.
# Usage: bash scripts/download_smolvlm.sh [quant]
#   quant: q4f16 (default, ~1.5GB), fp16 (~4GB), int8 (~3GB), bnb4 (~1.6GB)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
MODEL_DIR="$MODELS_DIR/SmolVLM-Instruct-onnx"

# The last commit before WebGPU format switch
COMMIT="4c2070c8361274f014d401a80ed7dcf3617a32bd"
BASE_URL="https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/resolve/$COMMIT"

QUANT="${1:-q4f16}"

case "$QUANT" in
  q4f16)
    DECODER="decoder_model_merged_q4f16.onnx"
    VISION="vision_encoder_q4f16.onnx"
    EMBED="embed_tokens_q4f16.onnx"
    ;;
  fp16)
    DECODER="decoder_model_merged_fp16.onnx"
    DECODER_DATA="decoder_model_merged_fp16.onnx_data"
    VISION="vision_encoder_fp16.onnx"
    EMBED="embed_tokens_fp16.onnx"
    ;;
  int8)
    DECODER="decoder_model_merged_int8.onnx"
    VISION="vision_encoder_int8.onnx"
    EMBED="embed_tokens_int8.onnx"
    ;;
  bnb4)
    DECODER="decoder_model_merged_bnb4.onnx"
    VISION="vision_encoder_bnb4.onnx"
    EMBED="embed_tokens_bnb4.onnx"
    ;;
  *)
    echo "Unknown quant: $QUANT (options: q4f16, fp16, int8, bnb4)"
    exit 1
esac

echo "Downloading SmolVLM-Instruct ONNX ($QUANT) ..."
echo "  Commit: $COMMIT"
echo "  Files: $DECODER, $VISION, $EMBED"
echo

mkdir -p "$MODEL_DIR/onnx"

# Download ONNX files
download() {
  local file="$1"
  local url="$BASE_URL/onnx/$file"
  local dest="$MODEL_DIR/onnx/$file"
  if [ -f "$dest" ] && [ -s "$dest" ]; then
    echo "  [SKIP] $file already exists"
    return
  fi
  echo "  [DOWNLOAD] $file ..."
  curl -sL --retry 3 "$url" -o "$dest"
  local size=$(stat -c%s "$dest" 2>/dev/null || echo 0)
  if [ "$size" -lt 1000 ]; then
    echo "  [ERROR] $file seems too small ($size bytes), download likely failed."
    return 1
  fi
  echo "  [OK] $file ($(numfmt --to=iec $size))"
}

download "$DECODER" || exit 1
if [ -n "$DECODER_DATA" ]; then
  download "$DECODER_DATA" || true
fi
download "$VISION" || exit 1
download "$EMBED" || exit 1

# Download model metadata files (use raw for small json/text files)
RAW_BASE="https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/raw/$COMMIT"
for f in config.json preprocessor_config.json processor_config.json \
         tokenizer.json tokenizer_config.json special_tokens_map.json \
         added_tokens.json chat_template.json vocab.json merges.txt \
         generation_config.json; do
  dest="$MODEL_DIR/$f"
  if [ -f "$dest" ] && [ -s "$dest" ]; then
    continue
  fi
  curl -sL "$RAW_BASE/$f" -o "$dest" 2>/dev/null
done

cat "$MODEL_DIR/config.json" 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
tc=d.get('text_config',{})
print(f'  Text: hidden={tc.get(\"hidden_size\")}, layers={tc.get(\"num_hidden_layers\")}, heads={tc.get(\"num_attention_heads\")}')
vc=d.get('vision_config',{})
print(f'  Vision: hidden={vc.get(\"hidden_size\")}, layers={vc.get(\"num_hidden_layers\")}, heads={vc.get(\"num_attention_heads\")}, patch={vc.get(\"patch_size\")}')
print(f'  Image seq len: {d.get(\"image_seq_len\")}')
print(f'  Image token ID: {d.get(\"image_token_id\")}')
" 2>/dev/null

echo
echo "[OK] Model downloaded to $MODEL_DIR"
echo "To run inference: python scripts/run_smolvlm.py -i /path/to/image.jpg"