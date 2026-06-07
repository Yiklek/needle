"""Tests for compiler.py — TileLang compilation orchestration."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "build")
sys.path.insert(0, "python")

import tilelang.language as TL

# Import compiler.py directly to avoid triggering needle/__init__.py
# which has hard dependencies on the compiled C++ backend (ndarray_backend_cpu).
_compiler_path = Path(__file__).resolve().parent.parent / "python" / "needle" / "dsl" / "compiler.py"
_compiler_spec = importlib.util.spec_from_file_location("needle.dsl.compiler", _compiler_path)
_compiler = importlib.util.module_from_spec(_compiler_spec)
_compiler_spec.loader.exec_module(_compiler)
compile_kernel = _compiler.compile_kernel
KernelArtifact = _compiler.KernelArtifact
ParamMeta = _compiler.ParamMeta


def _make_add_prim_func():
    @TL.prim_func
    def add_kernel(
        X: TL.Buffer((128,), "float32"),
        Y: TL.Buffer((128,), "float32"),
        Z: TL.Buffer((128,), "float32"),
    ):
        with TL.Kernel(128) as bx:
            Z[bx] = X[bx] + Y[bx]
    return add_kernel


def test_compile_kernel_metal_returns_artifact():
    prim_func = _make_add_prim_func()
    artifact = compile_kernel(prim_func, name="add", target="metal")
    assert isinstance(artifact, KernelArtifact)
    assert artifact.kernel_name == "add"
    assert artifact.target == "metal"
    assert len(artifact.source_hash) == 16
    assert "metal_stdlib" in artifact.kernel_source
    assert artifact.entry_point != ""
    assert len(artifact.params_meta) == 3
    assert artifact.cache_path is not None


def test_compile_kernel_metal_source_is_valid():
    prim_func = _make_add_prim_func()
    artifact = compile_kernel(prim_func, name="add", target="metal")
    source = artifact.kernel_source
    assert "kernel void" in source
    assert "metal_stdlib" in source
    assert "buffer(0)" in source
    assert "blockIdx" in source


def test_compile_kernel_params_meta():
    prim_func = _make_add_prim_func()
    artifact = compile_kernel(prim_func, name="add", target="metal")
    inputs = [p for p in artifact.params_meta if p.role == "input"]
    outputs = [p for p in artifact.params_meta if p.role == "output"]
    assert len(inputs) == 2
    assert len(outputs) == 1


def test_cache_hit():
    import time
    prim_func = _make_add_prim_func()
    # 清理缓存
    art1 = compile_kernel(prim_func, name="add", target="metal")
    if art1.cache_path and art1.cache_path.parent.exists():
        shutil.rmtree(art1.cache_path.parent, ignore_errors=True)

    art1 = compile_kernel(prim_func, name="add", target="metal")
    t0 = time.monotonic()
    art2 = compile_kernel(prim_func, name="add", target="metal")
    t1 = time.monotonic()
    assert art1.source_hash == art2.source_hash
    assert art1.kernel_source == art2.kernel_source
    assert art1.cache_path == art2.cache_path


def compile_deliverable():
    """交付件展示"""
    prim_func = _make_add_prim_func()
    artifact = compile_kernel(prim_func, name="add", target="metal")
    print("=" * 60)
    print("DELIVERABLE: TileLang -> Metal Compilation")
    print("=" * 60)
    print(f"Kernel name: {artifact.kernel_name}")
    print(f"Target: {artifact.target}")
    print(f"Source hash: {artifact.source_hash}")
    print(f"Entry point: {artifact.entry_point}")
    print(f"Cache path: {artifact.cache_path}")
    print()
    print("--- Metal Source Code ---")
    print(artifact.kernel_source)
    print("--- End Metal Source ---")
    print(f"\nParams ({len(artifact.params_meta)}):")
    for p in artifact.params_meta:
        print(f"  {p.name}: role={p.role}, index={p.index}, dtype={p.dtype}")
    manifest_path = artifact.cache_path.parent / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    print("\n--- manifest.json ---")
    print(json.dumps(manifest, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    test_compile_kernel_metal_returns_artifact()
    print("PASS: test_compile_kernel_metal_returns_artifact")
    test_compile_kernel_metal_source_is_valid()
    print("PASS: test_compile_kernel_metal_source_is_valid")
    test_compile_kernel_params_meta()
    print("PASS: test_compile_kernel_params_meta")
    test_cache_hit()
    print("PASS: test_cache_hit")
    compile_deliverable()
    print("\nALL TESTS PASSED")
