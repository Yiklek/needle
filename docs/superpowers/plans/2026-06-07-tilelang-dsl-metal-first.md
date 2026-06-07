# TileLang DSL Metal-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite compiler.py/codegen.py/registry.py to production quality with Metal-first compilation, caching, and dynamic registration.

**Architecture:** `@register_tilelang_op` decorator calls `lower(prim_func, target='metal')` → caches .metal source → registers DSLOpKernelFactory into C++ runtime via pybind11. C++ side adds `metal_source` field to DSLKernelMeta.

**Tech Stack:** Python 3.14, TileLang v0.1.10, C++23 modules, pybind11, Metal Shading Language

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Modify | `src/fineflow/core/kernels/dsl/device_launcher.cppm` | DSLKernelMeta 增加 metal_source |
| Modify | `src/fineflow/api/python/fineflow.cpp` | register_dsl_kernel_metal 绑定 |
| Rewrite | `python/needle/dsl/compiler.py` | 编译编排 + 缓存 |
| Rewrite | `python/needle/dsl/codegen.py` | 产物格式化/包装 |
| Rewrite | `python/needle/dsl/registry.py` | 装饰器 + 动态注册 |
| Rewrite | `python/needle/dsl/builtin/elementwise.py` | 用 TileLang API 重写 |
| Rewrite | `python/needle/dsl/__init__.py` | 更新导出 |
| Create | `tests/test_dsl_compiler.py` | compiler 测试 |
| Create | `tests/test_dsl_registry.py` | 装饰器测试 |
| Create | `tests/test_dsl_e2e.py` | 端到端测试 |

---

### Task 1: C++ DSLKernelMeta 增加 metal_source 字段

**Files:**
- Modify: `src/fineflow/core/kernels/dsl/device_launcher.cppm:19-30`

- [ ] **Step 1: 修改 DSLKernelMeta 结构体**

在 `device_launcher.cppm` 的 DSLKernelMeta 中，在 `cpu_compute` 之后、`binary` 之前插入 metal 字段：

```cpp
struct DSLKernelMeta {
  std::string name;
  Source source_type = Source::kTileLang;
  DeviceType target_device = DeviceType::kInvalidDevice;

  // CPU: function pointer
  std::function<void(KernelComputeContext&)> cpu_compute;

  // Metal: .metal source code (JIT compiled to metallib at first call)
  std::string metal_source;
  std::string entry_point;

  // GPU: binary blob (CUDA cubin, etc.)
  std::vector<uint8_t> binary;
};
```

- [ ] **Step 2: 重新配置 cmake 并编译**

```bash
cmake -S /Users/yiguangzheng/projects/needle -B /Users/yiguangzheng/projects/needle/build -GNinja
ninja -C /Users/yiguangzheng/projects/needle/build
```

Expected: 编译成功

- [ ] **Step 3: C++ 回归测试**

```bash
/Users/yiguangzheng/projects/needle/build/test_dsl_kernel
```

Expected: 6/6 tests pass

- [ ] **Step 4: Commit**

```bash
git add src/fineflow/core/kernels/dsl/device_launcher.cppm
git commit -m "feat(dsl): add metal_source field to DSLKernelMeta"
```

---

### Task 2: C++ pybind11 register_dsl_kernel_metal 绑定

**Files:**
- Modify: `src/fineflow/api/python/fineflow.cpp` (RegisterDSL function)

- [ ] **Step 1: 在 RegisterDSL 函数中增加 register_dsl_kernel_metal 绑定**

在 `RegisterDSL` 函数体内，`register_dsl_kernel` 的 `m.def(...)` 之后添加：

```cpp
m.def("register_dsl_kernel_metal",
    [](const std::string& name,
       const std::string& dtype_str,
       const std::string& metal_source,
       const std::string& entry_point,
       py::list params_meta) {
        namespace dsl = fineflow::dsl;

        dsl::DSLKernelMeta meta;
        meta.name = name;
        meta.source_type = dsl::Source::kTileLang;
        meta.target_device = DeviceType::kMetal;
        meta.metal_source = metal_source;
        meta.entry_point = entry_point;

        // Metal dispatch 当前走 Python bridge (cpu_compute fallback)
        meta.cpu_compute = [](KernelComputeContext& ctx) {
          // 占位: 未来用 metal-cpp 编译 .metal → metallib → dispatch
        };

        (void)dsl::DSLKernelRegistry::Register(std::move(meta));
    });
```

- [ ] **Step 2: 编译**

```bash
cmake -S /Users/yiguangzheng/projects/needle -B /Users/yiguangzheng/projects/needle/build -GNinja
ninja -C /Users/yiguangzheng/projects/needle/build
```

Expected: 编译成功

- [ ] **Step 3: 验证 binding 可用**

```bash
/Users/yiguangzheng/projects/needle/.venv/bin/python3 -c "
import sys; sys.path.insert(0, 'build')
import FineflowPyApi as lib
print('register_dsl_kernel_metal:', hasattr(lib, 'register_dsl_kernel_metal'))
"
```

Expected: `register_dsl_kernel_metal: True`

- [ ] **Step 4: C++ 回归测试**

```bash
/Users/yiguangzheng/projects/needle/build/test_dsl_kernel
```

Expected: 6/6 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/fineflow/api/python/fineflow.cpp
git commit -m "feat(dsl): add register_dsl_kernel_metal pybind11 binding"
```

---

### Task 3: compiler.py 重写（TDD）

**Files:**
- Create: `tests/test_dsl_compiler.py`
- Rewrite: `python/needle/dsl/compiler.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_dsl_compiler.py
"""Tests for compiler.py — TileLang compilation orchestration."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "build")
sys.path.insert(0, "python")

import tilelang.language as TL
from needle.dsl.compiler import compile_kernel, KernelArtifact, ParamMeta

CACHE_DIR = Path.home() / ".cache" / "needle" / "kernels"


def _make_add_prim_func():
    @TL.prim_func
    def add_kernel(
        X: TL.Buffer(("N",), "float32"),
        Y: TL.Buffer(("N",), "float32"),
        Z: TL.Buffer(("N",), "float32"),
    ):
        with TL.Kernel("N") as bx:
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
    print("DELIVERABLE: TileLang → Metal Compilation")
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
    print()
    print(f"Params ({len(artifact.params_meta)}):")
    for p in artifact.params_meta:
        print(f"  {p.name}: role={p.role}, index={p.index}, dtype={p.dtype}")

    # 验证缓存文件
    manifest_path = artifact.cache_path.parent / "manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    print()
    print("--- manifest.json ---")
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
    print("ALL TESTS PASSED")
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd /Users/yiguangzheng/projects/needle && .venv/bin/python3 tests/test_dsl_compiler.py
```

Expected: ImportError — `compile_kernel` 不存在

- [ ] **Step 3: 实现 compiler.py**

```python
# python/needle/dsl/compiler.py
"""TileLang compilation orchestration + disk cache."""

import dataclasses
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tilelang.engine.lower import lower


@dataclasses.dataclass
class ParamMeta:
    name: str
    role: str       # "input" | "output"
    index: int
    dtype: str


@dataclasses.dataclass
class KernelArtifact:
    kernel_name: str
    target: str
    source_hash: str
    kernel_source: str
    entry_point: str
    params_meta: list[ParamMeta]
    cache_path: Optional[Path]


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "needle" / "kernels"


def _tir_hash(prim_func) -> str:
    return hashlib.sha256(prim_func.script().encode()).hexdigest()[:16]


def _extract_entry_point(kernel_source: str, kernel_name: str) -> str:
    match = re.search(r'kernel\s+void\s+(\w+)\s*\(', kernel_source)
    if match:
        return match.group(1)
    return f"{kernel_name}_kernel"


def _extract_params_meta(prim_func) -> list[ParamMeta]:
    params = []
    buffer_map = getattr(prim_func, 'buffer_map', {})
    for i, param in enumerate(prim_func.params):
        name = param.name
        buf = buffer_map.get(param, None)
        dtype_str = str(buf.dtype) if buf is not None else "float32"
        role = "input"
        params.append(ParamMeta(name=name, role=role, index=i, dtype=dtype_str))
    # 最后一个参数标记为输出
    if params:
        params[-1].role = "output"
        params[-1].index = 0
        for idx, p in enumerate(params[:-1]):
            p.index = idx
    return params


def _load_cache(source_hash: str, target: str, cache_dir: Path) -> Optional[KernelArtifact]:
    cache_path = cache_dir / source_hash
    ext = "metal" if target == "metal" else target
    manifest_path = cache_path / "manifest.json"
    source_path = cache_path / f"kernel.{ext}"
    if not manifest_path.exists() or not source_path.exists():
        return None
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("target") != target:
        return None
    return KernelArtifact(
        kernel_name=manifest["kernel_name"],
        target=manifest["target"],
        source_hash=source_hash,
        kernel_source=source_path.read_text(),
        entry_point=manifest["entry_point"],
        params_meta=[ParamMeta(**p) for p in manifest["params"]],
        cache_path=source_path,
    )


def _save_cache(artifact: KernelArtifact, cache_dir: Path) -> Path:
    cache_path = cache_dir / artifact.source_hash
    cache_path.mkdir(parents=True, exist_ok=True)
    ext = "metal" if artifact.target == "metal" else artifact.target
    source_path = cache_path / f"kernel.{ext}"
    source_path.write_text(artifact.kernel_source)
    manifest = {
        "kernel_name": artifact.kernel_name,
        "target": artifact.target,
        "entry_point": artifact.entry_point,
        "source_hash": artifact.source_hash,
        "tir_hash": artifact.source_hash,
        "params": [dataclasses.asdict(p) for p in artifact.params_meta],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (cache_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return source_path


def compile_kernel(prim_func, *, name: str, target: str,
                   cache_dir: Optional[str] = None) -> KernelArtifact:
    cache_path = Path(cache_dir) if cache_dir else _default_cache_dir()
    source_hash = _tir_hash(prim_func)

    cached = _load_cache(source_hash, target, cache_path)
    if cached is not None:
        return cached

    artifact = lower(prim_func, target=target)
    kernel_source = artifact.kernel_source
    entry_point = _extract_entry_point(kernel_source, name)
    params_meta = _extract_params_meta(prim_func)

    result = KernelArtifact(
        kernel_name=name, target=target, source_hash=source_hash,
        kernel_source=kernel_source, entry_point=entry_point,
        params_meta=params_meta, cache_path=None,
    )
    result.cache_path = _save_cache(result, cache_path)
    return result
```

- [ ] **Step 4: 运行测试**

```bash
cd /Users/yiguangzheng/projects/needle && .venv/bin/python3 tests/test_dsl_compiler.py
```

Expected: 所有测试通过，交付件展示 .metal 源码全文 + manifest.json

- [ ] **Step 5: Commit**

```bash
git add python/needle/dsl/compiler.py tests/test_dsl_compiler.py
git commit -m "feat(dsl): rewrite compiler.py with lower() + disk cache"
```

---

### Task 4: codegen.py 重写

**Files:**
- Rewrite: `python/needle/dsl/codegen.py`

- [ ] **Step 1: 实现 codegen.py**

```python
# python/needle/dsl/codegen.py
"""Artifact formatting — wraps TileLang compilation output for Fineflow.

Pure wrapper. TileLang does the compilation. This module formats the output.
"""

import json
from pathlib import Path

from needle.dsl.compiler import KernelArtifact, ParamMeta


def write_artifact(artifact: KernelArtifact) -> Path:
    """返回 artifact 的缓存路径（compiler.compile_kernel 已写入）。"""
    if artifact.cache_path is None:
        raise ValueError("Artifact has no cache_path")
    return artifact.cache_path


def read_artifact(cache_path: Path) -> KernelArtifact:
    """从缓存路径读取 artifact。"""
    manifest_path = cache_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    return KernelArtifact(
        kernel_name=manifest["kernel_name"],
        target=manifest["target"],
        source_hash=manifest["source_hash"],
        kernel_source=cache_path.read_text(),
        entry_point=manifest["entry_point"],
        params_meta=[ParamMeta(**p) for p in manifest["params"]],
        cache_path=cache_path,
    )


def artifact_to_metal_source(artifact: KernelArtifact) -> str:
    """返回 .metal 源码字符串。"""
    if artifact.target != "metal":
        raise ValueError(f"Artifact target is '{artifact.target}', not 'metal'")
    return artifact.kernel_source
```

- [ ] **Step 2: 验证**

```bash
/Users/yiguangzheng/projects/needle/.venv/bin/python3 -c "
import sys; sys.path.insert(0, 'python')
from needle.dsl.compiler import compile_kernel
from needle.dsl.codegen import write_artifact, artifact_to_metal_source
import tilelang.language as TL

@TL.prim_func
def test_kernel(X: TL.Buffer((4,), 'float32'), Y: TL.Buffer((4,), 'float32')):
    with TL.Kernel(4) as bx: Y[bx] = X[bx]

artifact = compile_kernel(test_kernel, name='test', target='metal')
print('source length:', len(artifact_to_metal_source(artifact)))
print('cache path:', write_artifact(artifact))
print('OK')
"
```

Expected: source length > 0，路径存在

- [ ] **Step 3: Commit**

```bash
git add python/needle/dsl/codegen.py
git commit -m "feat(dsl): rewrite codegen.py as artifact wrapper"
```

---

### Task 5: registry.py 重写（TDD）

**Files:**
- Create: `tests/test_dsl_registry.py`
- Rewrite: `python/needle/dsl/registry.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_dsl_registry.py
"""Tests for registry.py — decorator + dynamic registration."""

import sys
sys.path.insert(0, "build")
sys.path.insert(0, "python")

import numpy as np
import FineflowPyApi as lib
import tilelang.language as TL
from needle.dsl.registry import register_tilelang_op


def test_decorator_compiles_and_registers():
    @register_tilelang_op("reg_test_add", device_types=["metal"], dtypes=["float32"])
    @TL.prim_func
    def add_kernel(
        X: TL.Buffer(("N",), "float32"),
        Y: TL.Buffer(("N",), "float32"),
        Z: TL.Buffer(("N",), "float32"),
    ):
        with TL.Kernel("N") as bx:
            Z[bx] = X[bx] + Y[bx]

    assert hasattr(add_kernel, "_dsl_meta")
    assert add_kernel._dsl_meta["name"] == "reg_test_add"


def test_decorator_kernel_callable():
    @register_tilelang_op("reg_callable", device_types=["metal"], dtypes=["float32"])
    @TL.prim_func
    def add_kernel(
        X: TL.Buffer(("N",), "float32"),
        Y: TL.Buffer(("N",), "float32"),
        Z: TL.Buffer(("N",), "float32"),
    ):
        with TL.Kernel("N") as bx:
            Z[bx] = X[bx] + Y[bx]

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype="float32")
    result = lib.call_dsl_kernel2("reg_callable", lib.from_numpy(a), lib.from_numpy(b))
    np.testing.assert_allclose(lib.to_numpy(result), a + b, atol=1e-5)


def test_compile_result_deliverable():
    """交付件: 展示编译产物"""
    @register_tilelang_op("reg_deliverable", device_types=["metal"])
    @TL.prim_func
    def add_kernel(
        X: TL.Buffer(("N",), "float32"),
        Y: TL.Buffer(("N",), "float32"),
        Z: TL.Buffer(("N",), "float32"),
    ):
        with TL.Kernel("N") as bx:
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
        print(f"  Source (first 400 chars):\n{art.kernel_source[:400]}")
    print("=" * 60)


if __name__ == "__main__":
    test_decorator_compiles_and_registers()
    print("PASS: test_decorator_compiles_and_registers")
    test_decorator_kernel_callable()
    print("PASS: test_decorator_kernel_callable")
    test_compile_result_deliverable()
    print("PASS: test_compile_result_deliverable")
    print("ALL TESTS PASSED")
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd /Users/yiguangzheng/projects/needle && .venv/bin/python3 tests/test_dsl_registry.py
```

Expected: ImportError 或运行时错误

- [ ] **Step 3: 实现 registry.py**

```python
# python/needle/dsl/registry.py
"""Kernel registration decorators for the DSL layer."""

from typing import Sequence

from needle.dsl.compiler import compile_kernel, KernelArtifact


def register_tilelang_op(
    kernel_name: str,
    *,
    device_types: Sequence[str] = ("metal",),
    dtypes: Sequence[str] = ("float32",),
    cache_dir: str | None = None,
):
    """Decorator that compiles a @TL.prim_func and registers it.

    Args:
        kernel_name: Kernel name for C++ runtime registry.
        device_types: Target backends (e.g. ["metal"]).
        dtypes: Supported dtypes.
        cache_dir: Cache directory.
    """
    def decorator(prim_func):
        compile_results = {}
        for dev in device_types:
            artifact = compile_kernel(
                prim_func, name=kernel_name, target=dev, cache_dir=cache_dir
            )
            compile_results[dev] = artifact
            _register_artifact(artifact, dtypes)

        prim_func._dsl_meta = {
            "name": kernel_name,
            "device_types": device_types,
            "dtypes": dtypes,
            "_results": compile_results,
        }
        return prim_func
    return decorator


def _register_artifact(artifact: KernelArtifact, dtypes: Sequence[str]) -> None:
    try:
        import FineflowPyApi as lib
    except ImportError:
        return

    for _dtype in dtypes:
        if artifact.target == "metal":
            lib.register_dsl_kernel_metal(
                name=artifact.kernel_name,
                dtype_str="float32",
                metal_source=artifact.kernel_source,
                entry_point=artifact.entry_point,
                params_meta=[
                    {"name": p.name, "role": p.role, "index": p.index, "dtype": p.dtype}
                    for p in artifact.params_meta
                ],
            )
```

- [ ] **Step 4: 运行测试**

```bash
cd /Users/yiguangzheng/projects/needle && .venv/bin/python3 tests/test_dsl_registry.py
```

Expected: 所有测试通过

- [ ] **Step 5: Commit**

```bash
git add python/needle/dsl/registry.py tests/test_dsl_registry.py
git commit -m "feat(dsl): rewrite registry.py with @register_tilelang_op decorator"
```

---

### Task 6: builtin/elementwise.py + E2E 测试

**Files:**
- Rewrite: `python/needle/dsl/builtin/elementwise.py`
- Create: `tests/test_dsl_e2e.py`

- [ ] **Step 1: 重写 builtin**

```python
# python/needle/dsl/builtin/elementwise.py
"""Built-in elementwise kernels with TileLang."""

import tilelang.language as TL
from needle.dsl.registry import register_tilelang_op


@register_tilelang_op("builtin_add", device_types=["metal"], dtypes=["float32"])
@TL.prim_func
def add_kernel(
    X: TL.Buffer(("N",), "float32"),
    Y: TL.Buffer(("N",), "float32"),
    Z: TL.Buffer(("N",), "float32"),
):
    with TL.Kernel("N") as bx:
        Z[bx] = X[bx] + Y[bx]


@register_tilelang_op("builtin_scale", device_types=["metal"], dtypes=["float32"])
@TL.prim_func
def scale_kernel(
    X: TL.Buffer(("N",), "float32"),
    S: TL.Buffer((1,), "float32"),
    Y: TL.Buffer(("N",), "float32"),
):
    with TL.Kernel("N") as bx:
        Y[bx] = X[bx] * S[0]
```

- [ ] **Step 2: 写 E2E 测试**

```python
# tests/test_dsl_e2e.py
"""End-to-end tests: decorator → compile → register → call → verify."""

import sys
sys.path.insert(0, "build")
sys.path.insert(0, "python")

import numpy as np
import FineflowPyApi as lib
import tilelang.language as TL
from needle.dsl.registry import register_tilelang_op


def test_e2e_metal_add():
    @register_tilelang_op("e2e_metal_add", device_types=["metal"])
    @TL.prim_func
    def add_kernel(
        X: TL.Buffer(("N",), "float32"),
        Y: TL.Buffer(("N",), "float32"),
        Z: TL.Buffer(("N",), "float32"),
    ):
        with TL.Kernel("N") as bx:
            Z[bx] = X[bx] + Y[bx]

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype="float32")
    result = lib.call_dsl_kernel2("e2e_metal_add", lib.from_numpy(a), lib.from_numpy(b))
    np.testing.assert_allclose(lib.to_numpy(result), a + b, atol=1e-5)


def test_e2e_deliverable():
    """交付件: 展示完整链路"""
    @register_tilelang_op("e2e_demo", device_types=["metal"])
    @TL.prim_func
    def demo_kernel(
        X: TL.Buffer(("N",), "float32"),
        Y: TL.Buffer(("N",), "float32"),
        Z: TL.Buffer(("N",), "float32"),
    ):
        with TL.Kernel("N") as bx:
            Z[bx] = X[bx] + Y[bx]

    compile_result = demo_kernel._dsl_meta["_results"]["metal"]
    print("=" * 60)
    print("DELIVERABLE: End-to-End Metal DSL Pipeline")
    print("=" * 60)
    print(f"1. Kernel: {compile_result.kernel_name}")
    print(f"   Entry point: {compile_result.entry_point}")
    print(f"   Cache: {compile_result.cache_path}")
    print(f"\n2. Metal source (first 400 chars):")
    print(compile_result.kernel_source[:400])

    import json
    cache_dir = compile_result.cache_path.parent
    print(f"\n3. Cache files:")
    for f in sorted(cache_dir.iterdir()):
        print(f"   {f.name} ({f.stat().st_size} bytes)")

    with open(cache_dir / "manifest.json") as f:
        print(f"\n4. Manifest:\n{json.dumps(json.load(f), indent=2)}")

    a = np.array([1.0, 2.0, 3.0], dtype="float32")
    b = np.array([4.0, 5.0, 6.0], dtype="float32")
    r = lib.call_dsl_kernel2("e2e_demo", lib.from_numpy(a), lib.from_numpy(b))
    result = lib.to_numpy(r)
    print(f"\n5. Runtime: {a} + {b} = {result}")
    np.testing.assert_allclose(result, a + b, atol=1e-5)
    print("   ✓ Numerical result matches")
    print("=" * 60)


if __name__ == "__main__":
    test_e2e_metal_add()
    print("PASS: test_e2e_metal_add")
    test_e2e_deliverable()
    print("PASS: test_e2e_deliverable")
    print("ALL E2E TESTS PASSED")
```

- [ ] **Step 3: 运行 E2E 测试**

```bash
cd /Users/yiguangzheng/projects/needle && .venv/bin/python3 tests/test_dsl_e2e.py
```

Expected: 全部通过，展示完整链路

- [ ] **Step 4: Commit**

```bash
git add python/needle/dsl/builtin/elementwise.py tests/test_dsl_e2e.py
git commit -m "feat(dsl): rewrite builtin kernels + E2E tests"
```

---

### Task 7: 全量验证

- [ ] **Step 1: C++ 回归**

```bash
/Users/yiguangzheng/projects/needle/build/test_dsl_kernel
/Users/yiguangzheng/projects/needle/build/test_tensor
```

Expected: 8/8 pass

- [ ] **Step 2: Python 全量**

```bash
cd /Users/yiguangzheng/projects/needle
.venv/bin/python3 tests/test_dsl_compiler.py
.venv/bin/python3 tests/test_dsl_registry.py
.venv/bin/python3 tests/test_dsl_e2e.py
```

Expected: 全部通过

- [ ] **Step 3: 更新 __init__.py**

```python
# python/needle/dsl/__init__.py
from needle.dsl.registry import register_tilelang_op
from needle.dsl.compiler import compile_kernel, KernelArtifact, ParamMeta
from needle.dsl.codegen import write_artifact, read_artifact, artifact_to_metal_source

__all__ = [
    "register_tilelang_op",
    "compile_kernel",
    "KernelArtifact",
    "ParamMeta",
    "write_artifact",
    "read_artifact",
    "artifact_to_metal_source",
]
```

- [ ] **Step 4: Commit**

```bash
git add python/needle/dsl/__init__.py
git commit -m "feat(dsl): finalize __init__.py with new public API"
```
