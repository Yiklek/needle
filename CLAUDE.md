# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See `AGENTS.md` for build/test/lint commands and code style conventions.

## Architecture overview

This is a deep learning framework with two layers:

**Python layer** (`python/needle/`): An autograd engine patterned after tinygrad/TensorFlow. `autograd.py` defines `Op`/`TensorOp`/`Value`/`Tensor` — each op has `compute()` (forward) and `gradient()` (backward). `ops.py` defines concrete ops (EWiseAdd, MatMul, etc.), `nn.py` has neural network modules, `optim.py` has optimizers. The `backend_selection.py` switches between backends via `NEEDLE_BACKEND` env var (`"nd"` for C++ ndarray, `"np"` for numpy).

**C++ layer** (`src/fineflow/`): High-performance backend using **C++23 named modules** (`.cppm` files). Built with CMake + custom `cmake/CppModules.cmake` and `cmake/CppModulesClang.cmake` (Clang-specific module support). Also has two traditional pybind11 modules (`src/ndarray_backend_cpu.cc` and `src/ndarray_backend_cuda.cu`) for the original assignment backend.

## C++ module dependency graph

```
FineflowCommon (common/*.cppm)
  ├── data_type_proto, device_type_proto, error_proto  (protobuf types)
  ├── registry_manager  (generic key-value registry + static registration)
  ├── data_type, hash, util, log, exception, fmt
  └── depends on: proto-objects, ThirdCommon (spdlog, fmt, expected)

FineflowCore (core/*.cppm + kernels/cpu/*.cppm + functional/*.cppm)
  ├── tensor.cppm       → Shape, Stride, ReadableTensorTrait, WritableTensorTrait, macros
  ├── blob_tensor.cppm   → Blob, AllocableBlobTensor<>, CpuTensor, BlobTensorView
  ├── op_kernel.cppm     → OpKernel base, KernelComputeContext
  ├── op_kernel_factory.cppm → Factory<>, OpKernelFactory, kernel registry types
  ├── functional.cppm    → Functor<R, Args...> base, FunctorTag registry
  ├── basic_functor.cppm → AddFunctor, FillFunctor, AssignFunctor, CompactFunctor + Call<> dispatch
  ├── kernels/cpu/*.cppm → Individual kernel impls (AddKernelImpl<T>, FillKernelImpl<T>, etc.)
  └── depends on: FineflowCommon

FineflowPyApiObj (api/python/*.cppm)
  ├── py_tensor.cppm    → Python-facing Tensor wrapper around BlobTensorView
  ├── py_functor.cppm   → Exposes functors to Python via pybind11
  └── depends on: FineflowCore (linked via WHOLE_ARCHIVE)

FineflowPyApi (api/python/fineflow.cpp)
  └── Traditional .cpp entry point, pybind11 module named "FineflowPyApi"
```

## C++ tensor class hierarchy

```
ReadableTensorTrait          (shape, stride, dtype, elementCount, isScalar)
  └── ReadableBlobTensorTrait  (+ bufferSize, offset, rawPtr, device)
        └── ReadableBlobTensor  (Blob storage, castPtr<T>)
              └── WritableBlobTensor  (+ shapeMut, strideMut, castPtrMut<T>, rawPtrMut)
                    ├── BlobTensor        (+ shared_from_this, view())
                    │     └── AllocableBlobTensor<Allocator>  (RAII alloc/free)
                    │           └── CpuTensor  (final, factory via CpuTensor::New())
                    └── BlobTensorView    (non-owning view: points to a BlobTensorPtr)
```

Key design decisions:
- `BlobTensor` destructor is **non-virtual** (performance, documented in code). Do NOT add virtual destructors or non-POD fields to subclasses.
- `BlobTensorView` is a copyable reference to a `shared_ptr<BlobTensor>` — it's the primary currency for passing tensors between ops.
- `AllocableBlobTensor` supports a `leak()` method to skip deallocation (used for Python interop).

## Registry/plugin system

The entire C++ layer uses a static-registration pattern:

1. **`RegistryMgr<Key, Value, Tag>`** — singleton map from Key to Value, parameterized by a Tag type for uniqueness.
2. **`RegisterTrigger<Key, Value, Tag>`** — static global variable that calls `RegistryMgr::Get().Register(key, value)` at construction time.
3. **`REGISTER_KEY_WITH_CLASS_T` macro** — creates a static `RegisterTrigger` in a translation unit.

Concrete registries:
- **Kernel factories**: `KernelFactoryRegistryMgr<AddKernel>` keyed by `DeviceType` → `OpKernelFactory` (template dispatch)
- **Runtime kernel factories**: `RuntimeKernelFactoryRegistryMgr` keyed by `{string_name, DeviceType}` → `OpKernelFactory` (string dispatch)
- **Functor registry**: `Functor<R, Args...>::RegistryFuncMgr` keyed by `string` → `std::function`
- **DataType size registry**: `DataTypeSizeRegistryMgr` keyed by `DataType` → `size_t`

## Kernel dispatch flow

When a functor (e.g., `AddFunctor::operator()`) is called:
1. Creates a `KernelComputeContext` with device and dtype
2. Inserts input/output tensors by name+index into the context
3. Calls `Call<AddKernel>(ctx)` or `Call("Add", ctx)`
4. `Call<T>` looks up `KernelFactoryRegistryMgr<T>::Get().GetValue(device)` → `OpKernelFactory`
5. Factory creates the kernel via `create(dtype)` → `std::unique_ptr<OpKernel>`
6. Kernel's `compute(ctx)` runs the actual operation, fetching tensors from context by name

Each kernel (e.g., `AddKernelImpl<T>`) is declared via `DECL_KERNEL(Add)` in the `.cppm` and its factory is implemented via `IMPL_KERNEL_FACTORY(Add)` which builds a dispatch map from DataType to specific template instantiations. The kernel is registered to both the typed and runtime registries via `REGISTER_KERNEL_FACTORY(Add, kCPU)`.

## Error handling convention (C++)

All functions that can fail return `Ret<T>` (alias for `tl::expected<T, Error>`). Use these macros:
- `CHECK_OR_RETURN(cond) << "message"` — return error if condition false
- `TRY(func_call)` — propagate error from a Ret-returning call
- `TRY_ASSIGN(var, func_call)` — assign result or propagate error

## Build targets

| Target | Type | Output |
|---|---|---|
| `ndarray_backend_cpu` | pybind11 module | `needle/backend_ndarray/ndarray_backend_cpu.so` (original assignment backend) |
| `ndarray_backend_cuda` | pybind11 module (CUDA) | `needle/backend_ndarray/ndarray_backend_cuda.so` |
| `FineflowPyApi` | pybind11 module | New C++23 modules-based Python API |
| `test_tensor` | CTest executable | C++ unit tests (gtest) |

## Key patterns to follow

- New C++ op = new kernel class (`DECL_KERNEL` + `IMPL_KERNEL_FACTORY` + `REGISTER_KERNEL_FACTORY` in `kernel_factor.h`) + new functor class + Python binding in `py_functor.cppm`
- Use `BlobTensorView` (not raw pointers) to pass tensors between ops
- Add new ops to `tests/test_fineflow_api.py` for integration testing and `tests/cpp/` for C++ unit tests
- After changing C++ code, run `make lib` to rebuild; after changing Python, `pip install -e .`
