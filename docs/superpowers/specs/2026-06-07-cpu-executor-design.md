# CPU Multi-Threaded Executor — Design Spec

**Date:** 2026-06-07
**Status:** Approved
**Target:** OpenCL/CUTLASS-style CPU kernel executor based on NVIDIA/stdexec, Actor model, TDD

## 1. Context

Current CPU kernel execution is single-threaded (CpuDeviceLauncher calls a lambda directly). Need a multi-threaded executor supporting NDRange-style work decomposition, multi-stage pipelines with cross-stage barriers, atomics, and concurrent streams. Based on `./stdexec` (NVIDIA Senders/Receivers, C++26 `std::execution`).

### MVP — What's Included

| Feature | Implementation |
|---------|---------------|
| Work-group / Work-item decomposition | stdexec `bulk()` + inner loop |
| WG barrier (`__syncthreads`) | `std::barrier<>(wg_size)` per work-group |
| Global atomics | `std::atomic<T>` with memory ordering |
| Memory fence (`__threadfence`) | `std::atomic_thread_fence` |
| Multi-stage pipeline with barriers | stdexec `let_value()` + `then()` chain |
| Streams (concurrent kernels) | Multiple stdexec sender pipelines |
| Dual kernel type (lambda + .dylib) | `std::function` + `dlopen`/`dlsym` |

### What's NOT Included (CPU doesn't need)

- Shared memory (`__shared__`) — CPU L1/L2/L3 cache handles data reuse automatically
- Warp shuffle / Warp vote — no SIMT warp concept on CPU
- Tensor Cores — BLAS libraries handle this
- Constant memory — compiler `const __restrict` already optimal

## 2. Architecture

```
┌──────────────────────────────────────────────────┐
│              CpuExecutor (调度层)                  │
│  build_graph() → stage().barrier().stage()...    │
│  → stdexec sender pipeline → 线程池分发            │
└────────────────────┬─────────────────────────────┘
                     │ stdexec bulk()
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Thread 0│   │ Thread 1│   │ Thread N│
│ Actor   │   │ Actor   │   │ Actor   │     Actor 线程池
└────┬────┘   └────┬────┘   └────┬────┘
     │             │             │
     ▼             ▼             ▼
  WG [0..N]     WG [...]    WG [...]
  wg_barrier()                work-group 内同步
  for wi in WG:
    kernel(global_id, wg_id, wi_id, args...)
```

## 3. Programming Model

```cpp
// Single stage dispatch
auto kernel = CpuKernel::FromLambda(
    [](int global_id, int wg_id, int wi_id, float* x, float* y, float alpha) {
        y[global_id] = x[global_id] * alpha;
    });

auto executor = CpuExecutor::Create(num_threads);
executor.enqueue(kernel, CpuRange::Global(1024), CpuRange::Local(128), x, y, alpha);

// Multi-stage with barriers
executor.build_graph()
    .stage("load",    kernel_load,    CpuRange::Global(1024), CpuRange::Local(128))
    .barrier()
    .stage("compute", kernel_compute, CpuRange::Global(1024), CpuRange::Local(128))
    .barrier()
    .stage("store",   kernel_store,   CpuRange::Global(1024), CpuRange::Local(128))
    .submit();

// Streams (concurrent execution)
auto s0 = executor.create_stream();
auto s1 = executor.create_stream();
s0.enqueue(kernel_a, ...);
s1.enqueue(kernel_b, ...);
```

## 4. Core Types

### 4.1 CpuRange

```cpp
struct CpuRange {
    size_t size;
    static CpuRange Global(size_t n) { return {n}; }
    static CpuRange Local(size_t n)  { return {n}; }
};
```

### 4.2 CpuKernel

```cpp
class CpuKernel {
public:
    template<typename F> static CpuKernel FromLambda(F&& fn);
    static CpuKernel FromLibrary(const std::string& path, const std::string& entry);
    void execute(int global_id, int wg_id, int wi_id, void** args);
};
```

### 4.3 CpuActor

```cpp
class CpuActor {
    std::barrier<> wg_barrier_;
    void execute_wg(CpuKernel& k, int wg_id, CpuRange global, CpuRange local, void** args);
};
```

### 4.4 CpuExecutor / CpuGraphBuilder

```cpp
class CpuExecutor {
    stdexec::static_thread_pool pool_;
    std::vector<CpuActor> actors_;
public:
    void enqueue(CpuKernel& k, CpuRange g, CpuRange l, Args... args);
    CpuGraphBuilder build_graph();
    CpuStream create_stream();
};

class CpuGraphBuilder {
    std::vector<stdexec::sender> stages_;
public:
    CpuGraphBuilder& stage(name, CpuKernel& k, CpuRange g, CpuRange l, Args...);
    CpuGraphBuilder& barrier();
    CpuGraphBuilder& memory_fence();
    void submit();
};
```

## 5. stdexec Pipeline (Multi-Stage)

```
schedule(pool)
  | bulk(N_wg, [kernel0](wg) { /* stage 0 */ })
  | then(cleanup)
  | let_value → bulk(N_wg, [kernel1](wg) { /* stage 1 */ })   ← implicit barrier
  | then(cleanup)
  | let_value → bulk(N_wg, [kernel2](wg) { /* stage 2 */ })
  | sync_wait()
```

Each `then()` + `let_value()` pair forms an implicit global barrier — stage N completes before stage N+1 starts.

## 6. Synchronization

| Primitive | Implementation | GPU Equivalent |
|-----------|---------------|----------------|
| WG barrier | `std::barrier<>(wg_size)` per work-group | `__syncthreads()` |
| Global atomics | `std::atomic<T>::fetch_add()` + memory_order | `atomicAdd()` |
| Memory fence | `std::atomic_thread_fence(seq_cst)` | `__threadfence()` |
| Stage barrier | stdexec `let_value()` chain | sequential launch |
| Streams | Multiple stdexec pipelines concurrently | CUDA streams |

## 7. Integration

```
NativeCpuDeviceLauncher::launch(meta, ctx)
  │
  ├─ dlopen(meta.native_lib_path)
  ├─ dlsym(meta.entry_point)
  ├─ Create CpuKernel::FromLibrary(handle, entry)
  ├─ Derive CpuRange from ctx tensor shape
  ├─ Pack tensor pointers → args array
  └─ executor.enqueue(kernel, global_range, local_range, args)
```

## 8. TDD Test Strategy

### C++ Unit Tests
- `CpuRange` construction
- `CpuKernel::FromLambda` single work-item
- `CpuActor` single work-group with `std::barrier`
- `CpuExecutor::enqueue` 1D kernel with N work-items
- `CpuGraphBuilder` 2-stage pipeline with barrier
- Atomics correctness (concurrent `fetch_add`)
- Memory fence (writer → reader fence ordering)

### Integration Tests
- Matrix multiply: 2-stage (load tile → compute tile)
- Reduction with atomics
- Concurrent streams (2 non-overlapping kernels)

### Performance
- Single-threaded vs multi-threaded element-wise add
- Strong scaling: 1/2/4/8 threads, 1M elements

## 9. Files

| Action | File | Purpose |
|--------|------|---------|
| Create | `src/fineflow/core/executor/cpu_range.cppm` | CpuRange |
| Create | `src/fineflow/core/executor/cpu_kernel.cppm` | CpuKernel (lambda + dylib) |
| Create | `src/fineflow/core/executor/cpu_actor.cppm` | CpuActor + WG barrier |
| Create | `src/fineflow/core/executor/cpu_executor.cppm` | CpuExecutor, CpuGraphBuilder, CpuStream |
| Modify | `CMakeLists.txt` | Add executor modules + stdexec dep |
| Modify | `src/fineflow/core/kernels/dsl/device_launcher.cppm` | NativeCpuDeviceLauncher → CpuExecutor |
| Create | `tests/cpp/test_cpu_executor.cpp` | C++ unit tests |
| Create | `tests/test_cpu_executor.py` | Python integration tests |
