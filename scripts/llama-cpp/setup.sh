#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_VERSION="b9357"
BASE_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_VERSION}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/../../third-party/llama-cpp"

FLAVORS=(
    "llama-${LLAMA_CPP_VERSION}-bin-ubuntu-x64.tar.gz|cpu"
    "llama-${LLAMA_CPP_VERSION}-bin-ubuntu-vulkan-x64.tar.gz|vulkan"
    "llama-${LLAMA_CPP_VERSION}-bin-ubuntu-rocm-7.2-x64.tar.gz|rocm"
)

mkdir -p "${TARGET_DIR}"

for entry in "${FLAVORS[@]}"; do
    IFS='|' read -r archive dest <<< "$entry"
    url="${BASE_URL}/${archive}"
    out_dir="${TARGET_DIR}/out/${dest}"

    if [ -d "${out_dir}" ]; then
        echo "[skip] ${dest} already exists at ${out_dir}"
        continue
    fi

    echo "[download] ${archive}"
    curl -fSL -o "${TARGET_DIR}/${archive}" "${url}"

    mkdir -p "${out_dir}"
    echo "[extract]  ${archive} -> ${out_dir}/"
    tar -xzf "${TARGET_DIR}/${archive}" -C "${out_dir}"
done

echo "[done] llama.cpp ${LLAMA_CPP_VERSION} ready under ${TARGET_DIR}/out/"
