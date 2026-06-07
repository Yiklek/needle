# TileLang DSL Integration — Production Design Spec

**Date:** 2026-06-07
**Status:** Approved
**Target:** Metal-first, CPU via Python bridge, CUDA when toolkit available

## 1. Context

Needle 当前仅支持手写 C++/CUDA kernel。集成 TileLang DSL 可以实现：
- 用 Python/TileLang 编写 kernel，一次编写多后端可用
- 用户可通过装饰器自定义算子
- TileLang 编译器负责代码生成，我们不重新实现编译器

### 已验证的 TileLang v0.1.10 能力

| 能力 | 状态 | 说明 |
|------|------|------|
| `@TL.prim_func` 装饰器 | ✅ 可用 | 产生 `PrimFunc` 对象 |
| `.script()` TIR 输出 | ✅ 可用 | 完整 TIR 文本 |
| `lower(prim_func, target='metal')` | ✅ 可用 | 产出 `.metal` 源码 |
| `lower(prim_func, target='cuda')` | ❌ 无 CUDA 工具链 | 有工具链的机器上可用 |
| `lower(prim_func, target='llvm')` | ❌ 无 LLVM 后端 | 需要重新编译 TileLang |
| `lower(prim_func, target='c')` | ❌ 未实现 | — |

## 2. Architecture

```
用户代码
  @register_tilelang_op("add", device_types=["metal"], dtypes=["float32","float16"])
  @TL.prim_func
  def add_kernel(X, Y, Z): ...

        │
        ▼
registry.py ───────────────────────────────────────┐
  装饰器入口，串联编译 → 缓存 → 动态注册              │
        │                                           │
        ▼                                           │
compiler.py ───────────────────────────────────┐    │
  调用 lower(prim_func, target) → artifact      │    │
  提取 kernel_source → 缓存到磁盘               │    │
        │                                       │    │
        ▼                                       │    │
codegen.py ────────────────────────────────┐    │    │
  包装器: 把 TileLang 产物格式化             │    │    │
  → manifest.json + kernel.metal            │    │    │
  → 不解析 TIR，不做编译                    │    │    │
                                            │    │    │
C++ Runtime (已有 + 扩展) ◄─────────────────┘────┘────┘
  DSLKernelMeta (增加 metal_source 字段)
  MetalDeviceLauncher (轻量 stub，实际 dispatch 走 Python bridge)
  RuntimeKernelFactoryRegistryMgr
  动态注册: pybind11 register_dsl_kernel_metal()
```

## 3. Module Design

### 3.1 compiler.py — 编译编排

**职责:** 接收 PrimFunc + target，调用 TileLang 编译/产出，管理缓存。

**公共 API:**
```python
def compile_kernel(prim_func, *, name: str, target: str,
                   cache_dir: str | None = None) -> KernelArtifact:
    """调用 TileLang lower() 编译 PrimFunc，缓存结果，返回 artifact。"""
```

**KernelArtifact 结构:**
```python
@dataclass
class KernelArtifact:
    kernel_name: str
    target: str                # "metal" | "cuda" | "cpu"
    source_hash: str           # TIR .script() 的 sha256[:16]
    kernel_source: str         # TileLang 生成的目标源码 (.metal / .cu / .c)
    entry_point: str           # kernel 入口函数名
    params_meta: list[ParamMeta]  # 参数列表
    cache_path: Path | None    # 缓存目录路径

@dataclass
class ParamMeta:
    name: str
    role: str       # "input" | "output"
    index: int
    dtype: str      # "float32" | "float16" | ...
```

**内部流程:**
1. 计算 TIR hash (`prim_func.script()` 的 sha256)
2. 检查 `~/.cache/needle/kernels/{hash}/manifest.json` 是否存在且 target 匹配
3. 命中缓存 → 返回 `KernelArtifact`
4. 未命中 → `lower(prim_func, target=target)` → 提取 `.kernel_source`
5. 提取 `params_meta`（从 `prim_func.params` + `prim_func.buffer_map`）
6. 写入缓存: `manifest.json` + `kernel.{metal/cu}`
7. 返回 `KernelArtifact`

**缓存目录结构:**
```
~/.cache/needle/kernels/{source_hash}/
  ├── manifest.json
  └── kernel.metal
```

**manifest.json:**
```json
{
  "kernel_name": "add",
  "target": "metal",
  "entry_point": "add_kernel_kernel",
  "source_hash": "a1b2c3d4e5f6a7b8",
  "tir_hash": "1122334455667788",
  "params": [
    {"name": "X", "role": "input",  "index": 0, "dtype": "float32"},
    {"name": "Y", "role": "input",  "index": 1, "dtype": "float32"},
    {"name": "Z", "role": "output", "index": 0, "dtype": "float32"}
  ],
  "created_at": "2026-06-07T17:00:00"
}
```

### 3.2 codegen.py — 产物格式化

**职责:** 把 TileLang 编译产物格式化为 Fineflow 可用格式。纯包装器，不解析 TIR。

**公共 API:**
```python
def write_artifact(artifact: KernelArtifact) -> Path:
    """将 artifact 写入缓存目录。返回缓存路径。"""

def read_artifact(cache_path: Path) -> KernelArtifact:
    """从缓存目录读取 artifact。"""

def artifact_to_metal_source(artifact: KernelArtifact) -> str:
    """返回 .metal 源码字符串。"""
```

### 3.3 registry.py — 注册装饰器

**职责:** 用户入口，串联编译→缓存→C++ 运行时注册。

**公共 API:**
```python
@register_tilelang_op(
    kernel_name: str,
    *,
    device_types: Sequence[str] = ("metal",),
    dtypes: Sequence[str] = ("float32",),
    cache_dir: str | None = None,
)
def decorator(prim_func): ...
```

**内部流程 (每 device_type × dtype 组合):**
1. 对每个 device_type 编译: `compiler.compile_kernel(prim_func, name=kernel_name, target=device_type)`
2. 缓存 artifact
3. 对每个 dtype: `_register_to_cpp_runtime(artifact, dtype)`

**动态注册到 C++:**
```python
def _register_to_cpp_runtime(artifact: KernelArtifact, dtype: str):
    lib.register_dsl_kernel_metal(
        name=artifact.kernel_name,
        dtype=dtype,
        metal_source=artifact.kernel_source,
        entry_point=artifact.entry_point,
        params_meta=artifact.params_meta,
    )
```

## 4. C++ Runtime Extension

### 4.1 DSLKernelMeta 扩展

```cpp
struct DSLKernelMeta {
  std::string name;
  Source source_type = Source::kTileLang;
  DeviceType target_device = DeviceType::kInvalidDevice;

  std::function<void(KernelComputeContext&)> cpu_compute;

  // Metal: .metal 源码
  std::string metal_source;
  std::string entry_point;

  // GPU 二进制
  std::vector<uint8_t> binary;
};
```

### 4.2 MetalDeviceLauncher (轻量 stub)

```cpp
class MetalDeviceLauncher final : public DeviceLauncher {
public:
  Ret<void> launch(const DSLKernelMeta& meta, KernelComputeContext& ctx) override {
    if (meta.cpu_compute) { meta.cpu_compute(ctx); return {}; }
    return UNIMPLEMENTED_ERROR;
  }
};
```

### 4.3 新增 pybind11 绑定

```cpp
m.def("register_dsl_kernel_metal",
    [](const std::string& name,
       const std::string& dtype_str,
       const std::string& metal_source,
       const std::string& entry_point,
       py::list params_meta) {
        dsl::DSLKernelMeta meta;
        meta.name = name;
        meta.source_type = dsl::Source::kTileLang;
        meta.target_device = DeviceType::kMetal;
        meta.metal_source = metal_source;
        meta.entry_point = entry_point;
        dsl::DSLKernelRegistry::Register(std::move(meta));
    });
```

## 5. Test Strategy (TDD)

### 5.1 compiler.py 测试

```python
def test_compile_add_kernel_metal():
    """编译 add kernel → metal target，验证 artifact"""
    artifact = compile_kernel(add_kernel, name="add", target="metal")
    assert artifact.kernel_name == "add"
    assert "metal_stdlib" in artifact.kernel_source

def test_cache_hit():
    """第二次编译应命中缓存"""

def test_compile_deliverable():
    """交付件: 打印 .metal 源码全文 + 缓存路径"""
```

### 5.2 registry.py 测试

```python
def test_decorator_registers_kernel():
    """@register_tilelang_op 后可在 runtime 查找到"""

def test_multi_dtype_deliverable():
    """交付件: 展示 registry 中所有注册 key"""
```

### 5.3 E2E 测试

```python
def test_end_to_end_metal_add():
    """装饰器定义 → 编译 → 注册 → 调用 → 数值验证"""
    result = lib.call_dsl_kernel2("e2e_add", t0, t1)
    np.testing.assert_allclose(lib.to_numpy(result), a + b, atol=1e-5)
```

## 6. Verification

1. `pytest tests/test_dsl_compiler.py -v` → compiler 测试通过
2. `pytest tests/test_dsl_registry.py -v` → 装饰器测试通过
3. `pytest tests/test_dsl_e2e.py -v` → 端到端测试通过
4. `./build/tests/test_dsl_kernel` → C++ 回归通过
5. 交付件: `.metal` 源码 + 缓存目录 + registry 内容

## 7. File List

| Action | File | Purpose |
|--------|------|---------|
| Rewrite | `python/needle/dsl/compiler.py` | 编译编排 + 缓存 |
| Rewrite | `python/needle/dsl/codegen.py` | 产物格式化/包装 |
| Rewrite | `python/needle/dsl/registry.py` | 装饰器 + 动态注册 |
| Rewrite | `python/needle/dsl/builtin/elementwise.py` | 用新 API 重写内置算子 |
| Modify | `src/fineflow/core/kernels/dsl/device_launcher.cppm` | DSLKernelMeta 增加 metal_source 字段 |
| Modify | `src/fineflow/api/python/fineflow.cpp` | 增加 register_dsl_kernel_metal 绑定 |
| Create | `tests/test_dsl_compiler.py` | compiler 测试 |
| Create | `tests/test_dsl_registry.py` | 装饰器测试 |
| Create | `tests/test_dsl_e2e.py` | 端到端测试 |
