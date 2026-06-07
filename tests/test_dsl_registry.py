"""Tests for registry.py -- decorator + dynamic registration."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, "build")
sys.path.insert(0, "python")

import numpy as np
import FineflowPyApi as lib
import tilelang.language as TL

# Import compiler.py directly to avoid triggering needle/__init__.py
_compiler_path = Path(__file__).resolve().parent.parent / "python" / "needle" / "dsl" / "compiler.py"
_compiler_spec = importlib.util.spec_from_file_location("needle.dsl.compiler", _compiler_path)
_compiler = importlib.util.module_from_spec(_compiler_spec)
sys.modules["needle.dsl.compiler"] = _compiler
_compiler_spec.loader.exec_module(_compiler)

# Import registry.py directly (it imports from needle.dsl.compiler which is now registered)
_registry_path = Path(__file__).resolve().parent.parent / "python" / "needle" / "dsl" / "registry.py"
_registry_spec = importlib.util.spec_from_file_location("needle.dsl.registry", _registry_path)
_registry = importlib.util.module_from_spec(_registry_spec)
_registry_spec.loader.exec_module(_registry)
register_tilelang_op = _registry.register_tilelang_op


def test_decorator_compiles_and_registers():
    @register_tilelang_op("reg_test_add", device_types=["metal"], dtypes=["float32"])
    @TL.prim_func
    def add_kernel(
        X: TL.Buffer((128,), "float32"),
        Y: TL.Buffer((128,), "float32"),
        Z: TL.Buffer((128,), "float32"),
    ):
        with TL.Kernel(128) as bx:
            Z[bx] = X[bx] + Y[bx]

    assert hasattr(add_kernel, "_dsl_meta")
    assert add_kernel._dsl_meta["name"] == "reg_test_add"


def _test_add_compute(inputs, outputs):
    import numpy as np
    np.add(inputs[("in", 0)], inputs[("in", 1)], out=outputs[("out", 0)])


def test_decorator_kernel_callable():
    @register_tilelang_op("reg_callable", device_types=["metal"], dtypes=["float32"])
    @TL.prim_func
    def add_kernel(
        X: TL.Buffer((128,), "float32"),
        Y: TL.Buffer((128,), "float32"),
        Z: TL.Buffer((128,), "float32"),
    ):
        with TL.Kernel(128) as bx:
            Z[bx] = X[bx] + Y[bx]

    # Test that compilation + metal source is correct
    assert add_kernel._dsl_meta["name"] == "reg_callable"
    art = add_kernel._dsl_meta["_results"]["metal"]
    assert art.kernel_source is not None
    assert "metal_stdlib" in art.kernel_source

    # Test CPU compute works (via direct registration)
    def add_cpu(inputs, outputs):
        np.add(inputs[("in", 0)], inputs[("in", 1)], out=outputs[("out", 0)])
    lib.register_dsl_kernel("reg_callable_cpu", "cpu", add_cpu)

    a = np.ones(128, dtype="float32") * 1.0
    b = np.ones(128, dtype="float32") * 2.0
    result = lib.call_dsl_kernel2("reg_callable_cpu", lib.from_numpy(a), lib.from_numpy(b))
    np.testing.assert_allclose(lib.to_numpy(result), a + b, atol=1e-5)


def test_compile_result_deliverable():
    """Deliverable display"""
    @register_tilelang_op("reg_deliverable", device_types=["metal"])
    @TL.prim_func
    def add_kernel(
        X: TL.Buffer((128,), "float32"),
        Y: TL.Buffer((128,), "float32"),
        Z: TL.Buffer((128,), "float32"),
    ):
        with TL.Kernel(128) as bx:
            Z[bx] = X[bx] + Y[bx]

    results = add_kernel._dsl_meta["_results"]
    print("=" * 60)
    print("DELIVERABLE: @register_tilelang_op results")
    print("=" * 60)
    for dev, art in results.items():
        print(f"\nTarget: {dev}")
        print(f"  Entry point: {art.entry_point}")
        print(f"  Cache: {art.cache_path}")
        print(f"  Source length: {len(art.kernel_source)}")
        print(f"  Source (first 500 chars):\n{art.kernel_source[:500]}")
    print("=" * 60)


if __name__ == "__main__":
    test_decorator_compiles_and_registers()
    print("PASS: test_decorator_compiles_and_registers")
    test_decorator_kernel_callable()
    print("PASS: test_decorator_kernel_callable")
    test_compile_result_deliverable()
    print("PASS: test_compile_result_deliverable")
    print("\nALL TESTS PASSED")
