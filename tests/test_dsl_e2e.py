"""End-to-end tests: decorator -> compile -> register -> call -> verify."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, "build")
sys.path.insert(0, "python")

import numpy as np
import FineflowPyApi as lib
import tilelang.language as TL

# Pre-load compiler and registry
_cp = Path("python/needle/dsl/compiler.py")
_cs = importlib.util.spec_from_file_location("needle.dsl.compiler", _cp)
_c = importlib.util.module_from_spec(_cs)
sys.modules["needle.dsl.compiler"] = _c
_cs.loader.exec_module(_c)

_rp = Path("python/needle/dsl/registry.py")
_rs = importlib.util.spec_from_file_location("needle.dsl.registry", _rp)
_r = importlib.util.module_from_spec(_rs)
_rs.loader.exec_module(_r)
register_tilelang_op = _r.register_tilelang_op


def test_e2e_metal_add():
    @register_tilelang_op("e2e_metal_add", device_types=["metal"])
    @TL.prim_func
    def add_kernel(
        X: TL.Buffer((128,), "float32"),
        Y: TL.Buffer((128,), "float32"),
        Z: TL.Buffer((128,), "float32"),
    ):
        with TL.Kernel(128) as bx:
            Z[bx] = X[bx] + Y[bx]

    results = add_kernel._dsl_meta["_results"]
    assert "metal" in results
    assert "metal_stdlib" in results["metal"].kernel_source

    # CPU compute test
    def add_cpu(inputs, outputs):
        np.add(inputs[("in", 0)], inputs[("in", 1)], out=outputs[("out", 0)])
    lib.register_dsl_kernel("e2e_metal_add", "cpu", add_cpu)

    a = np.ones(128, dtype="float32")
    b = np.ones(128, dtype="float32") * 2.0
    result = lib.call_dsl_kernel2("e2e_metal_add", lib.from_numpy(a), lib.from_numpy(b))
    np.testing.assert_allclose(lib.to_numpy(result), a + b, atol=1e-5)


def test_e2e_deliverable():
    """Full pipeline deliverable display"""
    @register_tilelang_op("e2e_demo", device_types=["metal"])
    @TL.prim_func
    def demo_kernel(
        X: TL.Buffer((128,), "float32"),
        Y: TL.Buffer((128,), "float32"),
        Z: TL.Buffer((128,), "float32"),
    ):
        with TL.Kernel(128) as bx:
            Z[bx] = X[bx] + Y[bx]

    compile_result = demo_kernel._dsl_meta["_results"]["metal"]
    print("=" * 60)
    print("DELIVERABLE: End-to-End Metal DSL Pipeline")
    print("=" * 60)
    print(f"1. Kernel: {compile_result.kernel_name}")
    print(f"   Entry: {compile_result.entry_point}")
    print(f"   Cache: {compile_result.cache_path}")
    print(f"\n2. Metal source:\n{compile_result.kernel_source[:500]}")

    import json
    cache_dir = compile_result.cache_path.parent
    print(f"\n3. Cache files:")
    for f in sorted(cache_dir.iterdir()):
        print(f"   {f.name} ({f.stat().st_size} bytes)")

    with open(cache_dir / "manifest.json") as f:
        print(f"\n4. Manifest:\n{json.dumps(json.load(f), indent=2)}")

    def add_cpu(inputs, outputs):
        np.add(inputs[("in", 0)], inputs[("in", 1)], out=outputs[("out", 0)])
    lib.register_dsl_kernel("e2e_demo", "cpu", add_cpu)

    a = np.array([1.0, 2.0, 3.0], dtype="float32")
    b = np.array([4.0, 5.0, 6.0], dtype="float32")
    r = lib.call_dsl_kernel2("e2e_demo", lib.from_numpy(a), lib.from_numpy(b))
    result = lib.to_numpy(r)
    print(f"\n5. Runtime: {a} + {b} = {result}")
    np.testing.assert_allclose(result, a + b, atol=1e-5)
    print("   OK - numerical result matches")
    print("=" * 60)


if __name__ == "__main__":
    test_e2e_metal_add()
    print("PASS: test_e2e_metal_add")
    test_e2e_deliverable()
    print("PASS: test_e2e_deliverable")
    print("\nALL E2E TESTS PASSED")
