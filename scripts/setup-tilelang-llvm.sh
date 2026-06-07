#!/bin/bash
# Setup TileLang with LLVM CPU backend for native .dylib codegen.
# Usage: bash scripts/setup-tilelang-llvm.sh
# Prerequisites: Homebrew LLVM (brew install llvm)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python3"

TILELANG_DIR="$PROJECT_DIR/../tilelang"
TILELANG_REPO="https://github.com/tile-ai/tilelang"
LLVM_PREFIX="$(brew --prefix llvm 2>/dev/null || echo /opt/homebrew/opt/llvm)"

echo "=== Step 1: Check prerequisites ==="
if [ ! -x "$LLVM_PREFIX/bin/llvm-config" ]; then
    echo "ERROR: LLVM not found at $LLVM_PREFIX. Install with: brew install llvm"
    exit 1
fi
echo "  LLVM: $LLVM_PREFIX ($($LLVM_PREFIX/bin/llvm-config --version))"
echo "  Python: $VENV_PYTHON"

echo ""
echo "=== Step 2: Clone TileLang ==="
if [ -d "$TILELANG_DIR" ]; then
    echo "  Already cloned at $TILELANG_DIR"
else
    git clone --depth 1 "$TILELANG_REPO" "$TILELANG_DIR"
    echo "  Cloned to $TILELANG_DIR"
fi

echo ""
echo "=== Step 3: Checkout TVM submodule ==="
cd "$TILELANG_DIR"
if [ ! -f 3rdparty/tvm/CMakeLists.txt ]; then
    git submodule update --init --depth 1 3rdparty/tvm
    echo "  TVM submodule initialized"
else
    echo "  TVM submodule already present"
fi

echo ""
echo "=== Step 4: Apply LLVM CPU patch ==="
PATCH_FILE="$PROJECT_DIR/docs/tilelang-llvm-cpu.patch"
if git diff --quiet tilelang/engine/lower.py 2>/dev/null; then
    if [ -f "$PATCH_FILE" ]; then
        git apply "$PATCH_FILE"
        echo "  Patch applied"
    else
        echo "  WARNING: patch file not found at $PATCH_FILE"
    fi
else
    echo "  Patch already applied (or local changes present)"
fi

echo ""
echo "=== Step 5: Create build config with LLVM enabled ==="
rm -rf build
mkdir -p build
cat > build/config.cmake << 'EOF'
# Enable LLVM backend for CPU native code generation
set(USE_LLVM ON)
EOF
echo "  build/config.cmake created"

echo ""
echo "=== Step 6: Install build dependencies ==="
"$VENV_PYTHON" -m pip install scikit_build_core cython z3-solver -q
echo "  Build dependencies installed"

echo ""
echo "=== Step 7: Build and install TileLang (editable) ==="
export PATH="$LLVM_PREFIX/bin:$PATH"
"$VENV_PYTHON" -m pip install -e . --no-build-isolation 2>&1 | tail -5
echo "  Build complete"

echo ""
echo "=== Step 8: Verify LLVM backend ==="
"$VENV_PYTHON" -c "
import tilelang.language as TL
from tilelang.engine.lower import lower

@TL.prim_func
def test(X: TL.Buffer((4,), 'float32'), Y: TL.Buffer((4,), 'float32')):
    with TL.Kernel(4, is_cpu=True) as bx:
        Y[bx] = X[bx] + X[bx]

art = lower(test, target='llvm', enable_device_compile=True)
import tempfile, os
d = tempfile.mkdtemp()
so = os.path.join(d, 'libtest.dylib')
art.rt_mod.export_library(so)
print(f'  Verified: .dylib exported ({os.path.getsize(so)} bytes)')
" && echo "  LLVM backend: OK" || echo "  LLVM backend: FAILED"

echo ""
echo "=== Done ==="
echo "TileLang with LLVM CPU backend installed at: $TILELANG_DIR"
echo "Usage: use T.Kernel(N, is_cpu=True) for CPU-compatible kernels"
