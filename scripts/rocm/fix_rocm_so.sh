#!/bin/bash
# Fix .so symlinks for pip-installed ROCm SDK (TheRock).
# Source this in your venv activate or run directly before using ROCm.
#
# Usage: source scripts/fix_rocm_so.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Find the .venv (try common locations)
if [ -n "$VIRTUAL_ENV" ]; then
    VENV="$VIRTUAL_ENV"
elif [ -d "$PROJECT_DIR/.venv" ]; then
    VENV="$PROJECT_DIR/.venv"
elif [ -d "$PROJECT_DIR/venv" ]; then
    VENV="$PROJECT_DIR/venv"
else
    echo "[fix_rocm] ERROR: Cannot find venv. Activate it first or set VIRTUAL_ENV."
    return 1 2>/dev/null || exit 1
fi

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES="$VENV/lib/python$PYVER/site-packages"

# ROCm SDK directories
CORE_LIB="$SITE_PACKAGES/_rocm_sdk_core/lib"
LIBS_DIRS=()
for d in "$SITE_PACKAGES"/_rocm_sdk_libraries_*/lib; do
    [ -d "$d" ] && LIBS_DIRS+=("$d")
done

ALL_LIB_DIRS=("$CORE_LIB")
for d in "${LIBS_DIRS[@]}"; do
    ALL_LIB_DIRS+=("$d")
done

fix_symlinks() {
    local dir=$1 count=0
    for f in "$dir"/lib*.so.*; do
        [ -f "$f" ] || continue
        base="${f%%.so*}.so"
        [ -e "$base" ] || [ -L "$base" ] && continue
        ln -sf "$(basename "$f")" "$base"
        ((count++))
    done
    return $count
}

create_alias() {
    local dir=$1 target=$2 actual=$3
    local t="$dir/$target"
    [ -e "$t" ] && return
    local a=$(find "$dir" -maxdepth 1 -name "$actual" | head -1)
    [ -z "$a" ] && return
    ln -sf "$(basename "$a")" "$t"
    echo "  alias: $target -> $(basename "$a")"
}

echo "[fix_rocm] Fixing .so symlinks..."
total=0
for d in "${ALL_LIB_DIRS[@]}"; do
    fix_symlinks "$d"
    total=$((total + $?))
done

# Fix subdirs
for d in "$CORE_LIB"/host-math/lib "$CORE_LIB"/rocm_sysdeps/lib; do
    fix_symlinks "$d"
done
echo "  $total symlinks created"

echo "[fix_rocm] Creating version aliases for onnxruntime-rocm compat..."
for libdir in "${ALL_LIB_DIRS[@]}"; do
    create_alias "$libdir" "libhipblas.so.2"     "libhipblas.so.3"
    create_alias "$libdir" "libamdhip64.so.6"    "libamdhip64.so.7"
    create_alias "$libdir" "librocm_smi64.so.7"  "librocm_smi64.so.1"
    create_alias "$libdir" "libroctracer64.so.4" "libroctracer64.so.4"
done

echo "[fix_rocm] Adding ROCm lib dirs to LD_LIBRARY_PATH..."
for d in "${ALL_LIB_DIRS[@]}"; do
    if [[ ":$LD_LIBRARY_PATH:" != *":$d:"* ]]; then
        export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
    fi
done

echo "[fix_rocm] Done. LD_LIBRARY_PATH includes $(echo ${#ALL_LIB_DIRS[@]}) ROCm lib dirs."