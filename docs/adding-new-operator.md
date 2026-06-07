# 如何新增算子

本文档说明在 Needle/Fineflow 框架中新增算子的完整流程，覆盖手写 C++ kernel 和 TileLang DSL kernel 两种方式。

## 目录

1. [架构概述](#架构概述)
2. [快速开始：DSL 算子](#快速开始dsl-算子)
3. [快速开始：手写 C++ 算子](#快速开始手写-c-算子)
4. [算子原型系统](#算子原型系统)
5. [属性系统](#属性系统)
6. [多后端支持](#多后端支持)
7. [测试规范](#测试规范)

---

## 架构概述

```
OpPrototype (声明)              Kernel Implementation (实现)
┌─────────────────┐           ┌──────────────────────────┐
│ name: "Conv"    │           │ C++: Conv2dKernelImpl    │
│ inputs: [X,W,B] │ ◄──────── │ DSL: conv2d_metal        │
│ outputs: [Y]    │  实现     │ Py:  conv2d_cpu_fn       │
│ attrs: {strides,│           └──────────────────────────┘
│   pads, group..}│
└─────────────────┘
```

**核心概念：先声明原型，再注册实现。** 原型定义算子的契约（输入/输出/属性），实现提供具体后端的执行代码。同一原型可以有多个实现（CPU/Metal/CUDA，手写/DSL）。

---

## 快速开始：DSL 算子

### 1. 用 TileLang 定义 kernel

```python
# python/needle/dsl/builtin/my_ops.py
import tilelang.language as TL
from needle.dsl.registry import register_tilelang_op

@register_tilelang_op(
    "gelu",                                    # kernel 名称
    device_types=["metal"],                    # 目标后端
    dtypes=["float32"],                        # 支持的 dtype
)
@TL.prim_func
def gelu_kernel(
    X: TL.Buffer((128,), "float32"),
    Y: TL.Buffer((128,), "float32"),
):
    with TL.Kernel(128) as bx:
        x = X[bx]
        c1 = TL.float32(0.044715)
        c2 = TL.float32(0.79788456)
        x3 = x * x * x
        inner = c2 * (x + c1 * x3)
        Y[bx] = TL.float32(0.5) * x * (TL.float32(1.0) + TL.tanh(inner))
```

### 2. 装饰器自动完成的步骤

1. 调用 `lower(prim_func, target='metal')` 生成 `.metal` 源码
2. 缓存到 `~/.cache/needle/kernels/{sha256}/kernel.metal`
3. 通过 pybind11 注册到 C++ `RuntimeKernelFactoryRegistryMgr`（key = `{name, kMetal}`）
4. 同时注册 CPU fallback（通过 Python bridge，key = `{name, kCPU}`）

```
~/.cache/needle/kernels/a00dddb80c6c5ccf/
├── manifest.json       ← 元数据（kernel名、entry point、参数列表）
└── kernel.metal         ← Metal Shading Language 源码
```

### 3. 调用 DSL kernel

```python
import FineflowPyApi as lib
import numpy as np

x = np.random.randn(128).astype("float32")
t_input = lib.from_numpy(x)

# 方式 A: call_dsl_kernel_v2 — 通用，支持任意输入输出
t_output = lib.Tensor(x.nbytes)
lib.call_dsl_kernel_v2("gelu", [t_input], [t_output])

# 方式 B: call_dsl_kernel — 单输入单输出
t_output = lib.call_dsl_kernel("gelu", t_input)

# 方式 C: call_dsl_kernel2 — 双输入单输出
# y = lib.call_dsl_kernel2("add", t_a, t_b)

result = lib.to_numpy(t_output)
```

### 4. 带属性的 DSL 算子

```python
@register_tilelang_op(
    "conv2d",
    device_types=["metal"],
    dtypes=["float32"],
)
@TL.prim_func
def conv2d_kernel(
    X: TL.Buffer(("N","C","H","W"), "float32"),
    W: TL.Buffer(("O","C","KH","KW"), "float32"),
    Y: TL.Buffer(("N","O","OH","OW"), "float32"),
):
    ...

# 属性在 compile_kernel 时以 attrs_schema 传入 DSLKernelMeta
# Python compute 函数通过第三个参数 attrs 接收：
def cpu_compute(inputs, outputs, attrs=None):
    strides = attrs.get("strides", [1, 1]) if attrs else [1, 1]
    pads = attrs.get("pads", [0, 0, 0, 0]) if attrs else [0, 0, 0, 0]
    ...
```

### 5. Python compute 函数签名

```python
# 旧签名（继续支持，无属性）
def compute(inputs: dict, outputs: dict):
    ...

# 新签名（推荐，支持属性 + 任意输入输出）
def compute(inputs: dict, outputs: dict, attrs: dict = None):
    ...
```

`inputs` / `outputs` 的 key 格式为 `("in", i)` / `("out", i)` 元组。

---

## 快速开始：手写 C++ 算子

### 1. 创建 kernel 文件

```cpp
// src/fineflow/core/kernels/cpu/gelu_kernel.cppm
module;
#include "fineflow/core/common/result.h"
#include "fineflow/core/common/util.h"
#include "fineflow/core/kernels/cpu/kernel_factor.h"

export module fineflow.core.kernels.cpu.gelu_kernel;

import fineflow.core.op_kernel;
import fineflow.core.blob_tensor;
import fineflow.core.common.device_type_proto;
import fineflow.core.common.data_type_proto;
import fineflow.core.common.registry_manager;
import fineflow.core.common.fmt;
import std;

export namespace fineflow {

// Step 1: DECL_KERNEL 声明 kernel 类和工厂
DECL_KERNEL(Gelu);

}  // namespace fineflow

// Step 2: 实现 kernel
namespace fineflow {

template <class T>
class GeluKernelImpl final : public GeluKernel {
  void compute(KernelComputeContext& ctx) const override {
    auto in0 = *ctx.fetchTensor("in", 0);
    auto out = *ctx.fetchTensor("out", 0);
    auto size = out.elementCount();

    const T* in_ptr = in0.castPtr<T>();
    T* out_ptr = out.castPtrMut<T>();

    // 读取运行时属性（可选）
    // auto& attrs = ctx.attrs();
    // double alpha = std::get<double>(attrs.at("alpha"));

    constexpr T c1 = T(0.044715);
    constexpr T c2 = T(0.7978845608028654);
    for (size_t i = 0; i < size; i++) {
      T x = in_ptr[i];
      T x3 = x * x * x;
      T inner = c2 * (x + c1 * x3);
      out_ptr[i] = T(0.5) * x * (T(1.0) + std::tanh(inner));
    }
  }
};

// Step 3: 生成 dtype → kernel 实现的 dispatch factory
IMPL_KERNEL_FACTORY(Gelu)

}  // namespace fineflow

// Step 4: 注册到 typed kernel registry
namespace fineflow {
namespace {
REGISTER_KERNEL_FACTORY(Gelu, DeviceType::kCPU);
}  // namespace
}  // namespace fineflow
```

### 2. 注册 functor

```cpp
// 在 src/fineflow/core/functional/basic_functor.cppm 中添加
REGISTER_FUNCTOR("gelu", Gelu);
```

### 3. 添加 Python binding

```cpp
// src/fineflow/api/python/fineflow.cpp
void RegisterGelu(py::module_& m) {
    auto gelu = std::function(
        PyFunctor<Tensor, const Tensor&>("gelu"));
    m.def("gelu", gelu);
}

// 在 PYBIND11_MODULE 中调用
PYBIND11_MODULE(PYBIND11_CURRENT_MODULE_NAME, m) {
    // ... 其他注册 ...
    RegisterGelu(m);
}
```

### 4. 确认 CMake 覆盖

`src/fineflow/core/kernels/cpu/*_kernel.cppm` 通过 glob 自动加入 FineflowCore，无需手动修改 CMakeLists.txt。

### 5. 调用

```python
import FineflowPyApi as lib
import numpy as np

x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="float32")
t = lib.from_numpy(x)
result = lib.gelu(t)  # 通过 functor → Call<GeluKernel> → GeluKernelImpl::compute
print(lib.to_numpy(result))
```

---

## 算子原型系统

原型定义算子的**契约**：名称、输入输出规格、属性 schema。一个原型可以有多个实现。

### OpPrototype 结构

```cpp
struct OpPrototype {
    std::string name;                            // "Conv"
    std::string domain = "";                     // "" = ai.onnx
    int64_t since_version = 1;
    std::vector<OpParam> inputs;
    std::vector<OpParam> outputs;
    std::unordered_map<std::string, AttrDef> attrs;
};

struct OpParam {
    std::string name;                            // "X", "W", "Y"
    enum class IoType { kInput, kOutput };
    IoType io_type;
    int64_t min_count = 1;                       // variadic 时可 > 1
    int64_t max_count = 1;                       // -1 = 无限 variadic
    bool optional = false;
};

struct AttrDef {
    AttrType type;
    AttrValue default_value;
};
```

### 注册原型

```cpp
auto proto = OpPrototype{
    .name = "Conv",
    .inputs = {
        OpParam{"X", OpParam::IoType::kInput},
        OpParam{"W", OpParam::IoType::kInput},
        OpParam{"B", OpParam::IoType::kInput, 1, 1, true},  // optional
    },
    .outputs = {OpParam{"Y", OpParam::IoType::kOutput}},
    .attrs = {
        {"kernel_shape", {AttrType::kInts, std::vector<int64_t>{}}},
        {"strides", {AttrType::kInts, std::vector<int64_t>{1, 1}}},
        {"pads", {AttrType::kInts, std::vector<int64_t>{0,0,0,0}}},
        {"group", {AttrType::kInt, int64_t(1)}},
    },
};
OpPrototypeRegistry::Get().Register("Conv", std::move(proto));
```

---

## 属性系统

### 属性类型

| AttrType | Python | C++ variant | 示例 |
|----------|--------|-------------|------|
| `kInt` | `int` | `int64_t` | `group=1`, `axis=0` |
| `kFloat` | `float` | `double` | `alpha=0.5`, `epsilon=1e-5` |
| `kString` | `str` | `std::string` | `auto_pad="SAME_UPPER"` |
| `kInts` | `list[int]` | `vector<int64_t>` | `strides=[2,2]` |
| `kFloats` | `list[float]` | `vector<double>` | 保留 |
| `kStrings` | `list[str]` | `vector<string>` | 保留 |

### C++ kernel 读取属性

```cpp
void compute(KernelComputeContext& ctx) const override {
    auto& attrs = ctx.attrs();
    auto& strides = std::get<std::vector<int64_t>>(attrs.at("strides"));
    auto group = std::get<int64_t>(attrs.at("group"));
    auto alpha = std::get<double>(attrs.at("alpha"));
}
```

### DSL Python compute 读取属性

```python
def cpu_compute(inputs, outputs, attrs=None):
    """attrs 是 dict，值为 python-native 类型"""
    strides = attrs.get("strides", [1, 1]) if attrs else [1, 1]
    group = attrs.get("group", 1) if attrs else 1
    ...
```

---

## 多后端支持

同一个算子名可以注册到不同 `DeviceType`，dispatch 时根据 `ctx.device()` 自动选择：

```
Call("Conv", ctx)  ← ctx.device() = kMetal
  │
  ▼
RuntimeKernelFactoryRegistryMgr::GetValue({"Conv", kMetal})
  │
  ▼
OpKernelFactory::create(dtype) → OpKernel::compute(ctx)
```

| 后端 | 注册方式 |
|------|---------|
| kCPU | 手写 `.cppm` + `REGISTER_KERNEL_FACTORY(Conv, kCPU)` |
| kCPU | DSL: `register_dsl_kernel(name, "cpu", fn)` (Python bridge) |
| kMetal | DSL: `register_dsl_kernel_metal(name, ..., metal_source, ...)` |
| kCUDA | DSL: `lower(prim_func, target='cuda')` (需要 CUDA 工具链) |

---

## 测试规范

### C++ kernel 测试

```cpp
// tests/cpp/test_gelu_kernel.cpp
TEST(GeluKernel, Compute) {
    auto a = CpuTensor::New(DataType::kFloat, Shape{5});
    a->castPtrMut<float>()[0] = -2.0f;
    a->castPtrMut<float>()[1] = -1.0f;
    a->castPtrMut<float>()[2] = 0.0f;
    a->castPtrMut<float>()[3] = 1.0f;
    a->castPtrMut<float>()[4] = 2.0f;

    auto c = CpuTensor::New(DataType::kFloat, Shape{5});

    KernelComputeContext ctx(DeviceType::kCPU, DataType::kFloat);
    ctx.insertTensor("in", 0, a->view());
    ctx.insertTensor("out", 0, c->view());

    // 如果有属性:
    // ctx.setAttrs({{"alpha", double(1.0)}});

    Call<GeluKernel>(ctx);  // 或 Call("gelu", ctx)

    EXPECT_NEAR(c->castPtr<float>()[2], 0.0f, 1e-5);
}
```

### Python DSL 测试

```python
# tests/test_gelu.py
import numpy as np
import FineflowPyApi as lib

def test_gelu_dsl():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="float32")
    t_input = lib.from_numpy(x)
    t_output = lib.Tensor(x.nbytes)
    lib.call_dsl_kernel_v2("gelu", [t_input], [t_output])
    actual = lib.to_numpy(t_output)

    # 参考 PyTorch
    import torch
    expected = torch.nn.functional.gelu(torch.tensor(x)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-4)
```

### Python 手写 kernel 测试

```python
# tests/test_fineflow_api.py
def test_gelu_handwritten():
    x = np.random.randn(128).astype("float32")
    t = lib.from_numpy(x)
    result = lib.gelu(t)
    actual = lib.to_numpy(result)
    # ... 验证
```

---

## 检查清单

- [ ] **原型**: `OpPrototype` 定义输入输出和属性 schema
- [ ] **实现**: 手写 `.cppm` 或 DSL `@register_tilelang_op`
- [ ] **CMake**: 手写 kernel 被 `*_kernel.cppm` glob 覆盖
- [ ] **Functor**: 手写 kernel 注册 functor
- [ ] **Binding**: 手写 kernel 添加 `m.def("xxx", ...)`
- [ ] **C++ 测试**: `tests/cpp/test_xxx.cpp`
- [ ] **Python 测试**: `tests/test_xxx.py`
- [ ] **属性**: ctx.attrs() 或 compute_fn(inputs, outputs, attrs) 正确处理
