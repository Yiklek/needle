# OpPrototype + Attr System — Design Spec

**Date:** 2026-06-07
**Status:** Approved
**Target:** OpPrototype registry, Attr type system, unified variable-args kernel call

## 1. Problem

- DSL kernels fixed at 1-2 inputs, 1 output (hardcoded `call_dsl_kernel`/`call_dsl_kernel2`)
- No operator attributes — can't express Conv stride/pads, Pooling kernel_shape, etc.
- No prototype → implementation contract or validation
- Hand-written C++ kernels can't receive attrs either

## 2. Architecture

```
OpPrototype (声明)              Kernel Implementation (实现)
┌─────────────────┐           ┌──────────────────────────┐
│ name: "Conv"    │           │ C++: Conv2dKernelImpl    │
│ inputs: [X,W,B] │ ◄──────── │ DSL: conv2d_metal        │
│ outputs: [Y]    │  实现     │ Py:  conv2d_cpu_fn       │
│ attrs: {strides,│           │                          │
│   pads, group..}│           └──────────────────────────┘
└──────┬──────────┘
       │
       ▼
OpPrototypeRegistry
  key: (name, domain)
       │
       ▼
RuntimeKernelFactoryRegistryMgr (已有)
  key: (name, device_type)
       │
       ▼
KernelComputeContext
  + setAttrs(AttrMap)      ← 新增
  + attrs() → AttrMap      ← 新增
```

## 3. Attr Type System

### 3.1 C++ (`op_kernel.cppm`)

```cpp
enum class AttrType { kInt, kFloat, kString, kInts, kFloats, kStrings };

using AttrValue = std::variant<
  int64_t, double, std::string,
  std::vector<int64_t>, std::vector<double>, std::vector<std::string>
>;

using AttrMap = std::unordered_map<std::string, AttrValue>;
```

### 3.2 Python ↔ C++ Mapping

| Python | C++ |
|--------|-----|
| `int` | `int64_t` |
| `float` | `double` |
| `str` | `std::string` |
| `list[int]` | `std::vector<int64_t>` |
| `list[float]` | `std::vector<double>` |
| `list[str]` | `std::vector<std::string>` |

## 4. OpPrototype

### 4.1 C++ Module (`op_prototype.cppm`)

```cpp
struct OpParam {
  std::string name;
  enum class IoType { kInput, kOutput };
  IoType io_type;
  int64_t min_count = 1;
  int64_t max_count = 1;  // -1 = unlimited (variadic)
  bool optional = false;
};

struct AttrDef {
  AttrType type;
  AttrValue default_value;
};

struct OpPrototype {
  std::string name;
  std::string domain = "";
  int64_t since_version = 1;
  std::vector<OpParam> inputs;
  std::vector<OpParam> outputs;
  std::unordered_map<std::string, AttrDef> attrs;
};

using OpPrototypeRegistry = RegistryMgr<std::string, OpPrototype, PrototypeTag>;
```

### 4.2 Python Decorator

```python
@register_op_prototype(
    name="Conv",
    inputs=[
        OpParam("X", io="input"),
        OpParam("W", io="input"),
        OpParam("B", io="input", optional=True),
    ],
    outputs=[OpParam("Y", io="output")],
    attrs={
        "kernel_shape": (AttrType.INTS, []),
        "strides": (AttrType.INTS, [1, 1]),
        "pads": (AttrType.INTS, [0, 0, 0, 0]),
        "group": (AttrType.INT, 1),
    },
)
```

## 5. KernelComputeContext Extension

```cpp
class KernelComputeContext {
public:
  void setAttrs(AttrMap attrs) { attrs_ = std::move(attrs); }
  const AttrMap& attrs() const { return attrs_; }

  // 已有 API 不变
  Ret<BlobTensorView> fetchTensor(...);
  void insertTensor(...);
  DeviceType device();
  DataType dtype();

private:
  AttrMap attrs_;  // 新增
  // ... existing fields ...
};
```

## 6. Unified Python Binding

### 6.1 call_dsl_kernel_v2 (variable inputs/outputs + attrs)

```cpp
m.def("call_dsl_kernel_v2",
    [](const std::string& name, py::list inputs, py::list outputs, py::dict attrs) {
        // 1. Build context
        // 2. Iterate inputs: ctx.insertTensor("in", i, tensor)
        // 3. Iterate outputs: ctx.insertTensor("out", i, tensor)
        // 4. ctx.setAttrs(convert(attrs))
        // 5. Call(name, ctx)
    });
```

### 6.2 Variable-length compute wrapper

```cpp
auto cpp_compute = [compute_fn](KernelComputeContext& ctx) {
    py::gil_scoped_acquire gil;
    py::dict inputs, outputs;

    for (size_t i = 0; ; i++) {
        auto t = ctx.fetchTensor("in", i);
        if (!t.has_value()) break;
        inputs[py::make_tuple("in", i)] = BlobViewToNumpy(t.value());
    }
    for (size_t i = 0; ; i++) {
        auto t = ctx.fetchTensor("out", i);
        if (!t.has_value()) break;
        outputs[py::make_tuple("out", i)] = BlobViewToNumpy(t.value());
    }

    // Convert attrs to Python dict
    py::dict py_attrs = ConvertAttrsToPython(ctx.attrs());

    compute_fn(inputs, outputs, py_attrs);
};
```

### 6.3 Python compute signature

```python
# 新签名（attrs 可选）
def cpu_compute(inputs: dict, outputs: dict, attrs: dict = None):
    strides = attrs.get("strides", [1, 1]) if attrs else [1, 1]
    # ...
```

## 7. DSLKernelMeta Extension

```cpp
struct DSLKernelMeta {
  // ... existing ...
  std::string metal_source;
  std::string entry_point;
  
  AttrMap attrs_schema;  // 新增: compile-time attribute defaults
};
```

## 8. Files

| Action | File |
|--------|------|
| Create | `src/fineflow/core/common/op_prototype.cppm` |
| Modify | `src/fineflow/core/op_kernel.cppm` |
| Modify | `src/fineflow/core/kernels/dsl/device_launcher.cppm` |
| Modify | `src/fineflow/core/kernels/dsl/dsl_kernel.cppm` |
| Modify | `src/fineflow/api/python/fineflow.cpp` |
| Modify | `python/needle/dsl/registry.py` |
| Create | `tests/test_dsl_attrs.py` |
| Modify | `tests/cpp/test_dsl_kernel.cpp` |

## 9. Test Strategy

- C++: `KernelComputeContext::setAttrs/attrs()` round-trip, `AttrValue` variant
- Python: `call_dsl_kernel_v2` with variable inputs (0/1/3/5), with/without attrs
- Regression: compiler/registry/E2E tests unchanged
