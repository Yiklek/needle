# Triton / TileLang Kernel DSL 集成方案

本文档设计如何将 Triton 和 TileLang 作为 DSL（Domain-Specific Language）集成到 Needle / Fineflow 的 kernel 系统中，支持开发内置算子和用户自定义算子。

## 一、背景与动机

当前 Fineflow 的内置 kernel 全部为手写 C++/CUDA，存在以下问题：

1. **开发效率低**：每新增一个算子需要写大量样板代码（`DECL_KERNEL` + `IMPL_KERNEL_FACTORY` + `REGISTER_KERNEL_FACTORY` + functor + Python 绑定）
2. **多后端成本高**：CUDA/Metal/Vulkan 等各后端需独立实现，工作量大且容易不一致
3. **用户无法自定义算子**：当前没有暴露算子开发接口给框架使用者
4. **性能调优困难**：手写 CUDA kernel 的 tiling/shared memory/warp scheduling 等优化门槛极高

引入 Triton 和 TileLang 作为 DSL 可同时解决以上问题：

- Triton：Python 编写 GPU kernel，~25 行 = 数百行 CUDA，生态成熟
- TileLang：Tile 级抽象 + 编译器自动推理 + 多后端完备（CUDA/ROCm/Metal/Vulkan/WebGPU/昇腾/算能/摩尔线程）

### TileLang 核心能力

| 维度 | 说明 |
|------|------|
| **Tile 级抽象** | 将调度空间（线程绑定、内存布局、Tensorize、流水线排布）与计算数据流解耦 |
| **三级编程模型** | Beginner（Auto Schedule）→ Developer（Tile 级手动）→ Expert（PTX/ASM） |
| **编译器自动推理** | Layout Inference（Strict→Common→Free）、Pipeline Inference、指令选择 Inference |
| **多后端** | CUDA/ROCm/Metal/Vulkan/WebGPU/昇腾/算能/摩尔线程 |
| **关键原语** | `T.Parallel`、`T.copy`、`T.GEMM`、`T.gemm_sp`（稀疏）、`T.reduce`、`T.Pipelined`、`T.tma_copy` |
| **生态采用** | DeepSeek V3.2/V4（算子快速原型）、微软 BitNet/BitBLAS、摩尔线程 TileLang-MUSA |
| **性能** | GEMM 与 CUTLASS 相当；Flash MLA 仅 80 行达 H100 95% 性能；AMD MI300X FlashMLA 与手写汇编持平 |
| **TileRT 运行时** | 专为大模型低延迟推理，B200 端到端延迟降 ~35% |

---

## 二、集成架构总览

### 2.1 整体分层

```
    ┌───────────────────────────────────────────────────┐
    │         needle.dsl (Python DSL layer)              │
    │  @register_triton_op / @register_tilelang_op       │
    │  — 面向用户的 kernel 注册 API                      │
    │  — 内置算子: needle/dsl/builtin/*.py              │
    │  — 用户自定义: 项目任意位置                           │
    └────────────────────┬──────────────────────────────┘
                         │
    ┌────────────────────▼──────────────────────────────┐
    │         KernelCompiler (C++ compilation infra)     │
    │  compile_from_triton(source, target_device)        │
    │  compile_from_tilelang(source, target_device)      │
    │  → 返回 KernelBinary + LaunchMeta                  │
    └────────────────────┬──────────────────────────────┘
                         │
    ┌────────────────────▼──────────────────────────────┐
    │         DSLOpKernel (C++ runtime adapter)          │
    │  封装任意已编译 kernel 二进制                       │
    │  compute(ctx): load params → launch → sync         │
    └────────────────────┬──────────────────────────────┘
                         │
    ┌────────────────────▼──────────────────────────────┐
    │         Existing Fineflow Registry                 │
    │  RuntimeKernelFactoryRegistryMgr                  │
    │  Key: {kernel_name, DeviceType}                   │
    │  Value: DSLOpKernelFactory                         │
    │  — 与手写 C++ kernel 共享同一 registry             │
    └───────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  CUDA    │  │  Metal   │  │  ROCm    │
    │  launcher│  │  launcher│  │  launcher│
    └──────────┘  └──────────┘  └──────────┘
```

### 2.2 与现有 C++ kernel 的关系

DSL kernel 和手写 C++ kernel 共存于同一 registry，互不冲突：

```
RuntimeKernelFactoryRegistryMgr:
  {"Add", kCPU}          → AddKernelFactory          (hand-written C++)
  {"Add", kCUDA}         → AddKernelFactory (CUDA)    (hand-written CUDA)
  {"FlashAttn", kCUDA}   → DSLOpKernelFactory        (TileLang-generated)
  {"MatMul", kCUDA}      → DSLOpKernelFactory        (Triton-generated)
  {"FlashAttn", kMetal}  → DSLOpKernelFactory        (TileLang-generated, same source)
  {"FlashAttn", kROCm}   → DSLOpKernelFactory        (TileLang-generated, same source)
  {"FusedGeLU", kCUDA}   → DSLOpKernelFactory        (Triton-generated, user-defined)
```

上层 `Call("Add", ctx)` 逻辑完全不变。对 Functor 和 Python 调用方完全透明。

---

## 三、Triton 集成方案

### 3.1 Triton Kernel 编译流程

```
  Python: @triton_op("my_add")
  def my_add_kernel(x_ptr, y_ptr, out_ptr, N,
                     BLOCK: tl.constexpr)
       │
       ▼ (1) JIT compile via triton.compile()
  ┌───────────────────────────┐
  │  TritonAOTCompiler         │
  │  → triton.compile()        │
  │  → cubin / PTX + metadata  │
  │  → cache to disk or embed  │
  └───────────┬───────────────┘
              │ (2) register as factory in registry
  ┌───────────▼───────────────┐
  │  TritonKernelFactory       │
  │  implements OpKernelFactory│
  │  create(dtype) →           │
  │    TritonOpKernel(meta)    │
  └───────────┬───────────────┘
              │ (3) launch at runtime
  ┌───────────▼───────────────┐
  │  TritonOpKernel            │
  │  compute(ctx):             │
  │    1. extract tensors      │
  │    2. set grid from meta   │
  │    3. cuLaunchKernel()     │
  │    4. wait for completion  │
  └───────────────────────────┘
```

### 3.2 C++ 侧核心组件

```cpp
// src/fineflow/core/kernels/dsl/dsl_kernel.cppm

// 编译后的 kernel 元数据
struct DSLKernelMeta {
  std::string name;               // kernel 入口函数名
  dsl::Source source_type;        // triton / tilelang
  std::vector<uint8_t> binary;    // 设备特定二进制 (cubin / metallib / etc.)
  DeviceType target_device;
  // launch 参数
  size_t num_warps;
  size_t shared_memory_bytes;
  std::function<GridConfig(const KernelComputeContext&)> grid_computer;
  std::function<std::vector<void*>(const KernelComputeContext&)> arg_extractor;
};

// 统一的 DSL kernel wrapper
class DSLOpKernel final : public OpKernel {
public:
  explicit DSLOpKernel(DSLKernelMeta meta, DeviceLauncher launcher);

  void compute(KernelComputeContext& ctx) const override {
    auto grid = meta_.grid_computer(ctx);
    auto args = meta_.arg_extractor(ctx);
    launcher_.launch(meta_.binary, meta_.entry_point,
                     grid, {meta_.num_warps * 32},
                     meta_.shared_memory_bytes, args, stream_);
  }
private:
  DSLKernelMeta meta_;
  DeviceLauncher launcher_;
};

// 工厂
class DSLOpKernelFactory final : public OpKernelFactory {
public:
  Ret<std::unique_ptr<OpKernel>> create(DataType) override;
private:
  DSLKernelMeta meta_;
};
```

### 3.3 Python 侧注册 API

```python
# python/needle/dsl/triton_ops.py

import triton
import triton.language as tl
from needle.dsl.registry import register_triton_op

@register_triton_op("add", device_types=["cuda"], dtypes=["float32", "float16"])
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N,
               BLOCK_SIZE: tl.constexpr = 1024):
    """Element-wise addition."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@register_triton_op("gelu", device_types=["cuda"], dtypes=["float32", "float16"])
@triton.jit
def gelu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr = 1024):
    """GELU activation."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    c1 = 0.044715
    c2 = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c2 * (x + c1 * x3)
    out = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(out_ptr + offsets, out, mask=mask)


@register_triton_op("fused_gelu_mul", device_types=["cuda"], dtypes=["float16"])
@triton.jit
def fused_gelu_mul_kernel(x_ptr, w_ptr, out_ptr,
                           M, N, K,
                           BLOCK_M: tl.constexpr,
                           BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr):
    """Fused: GELU(x @ W) * (x @ W) — 自定义融合算子."""
    ...
```

`@register_triton_op` 装饰器自动完成：

1. 调用 `triton.compile()` 编译出 cubin/PTX
2. 生成 `DSLKernelMeta`（含入口名、grid 计算、参数提取函数）
3. 调用 C++ 侧 `RegisterDSLKernel()` 将工厂注册到 runtime registry
4. 同时注册对应的 `Functor`，在 Python 层可通过统一接口调用

### 3.4 AOT vs JIT

| 模式 | 时机 | 缓存位置 | 适用场景 |
|------|------|----------|----------|
| **AOT（编译期）** | `setup.py build` 或 cmake 阶段 | 嵌入 C++ binary 或独立 `.bin` 文件 | 内置算子、发布版本 |
| **JIT（运行时）** | 首次调用 | `~/.needle/kernel_cache/` 或项目 `.needle_cache/` | 用户自定义算子、研究探索 |

建议策略：
- 内置算子 → AOT 编译，确保确定性和加载速度
- 用户自定义 → JIT + 缓存，保留灵活性
- 提供 `needle dsl freeze` 命令将 JIT 缓存提升为 AOT

---

## 四、TileLang 集成方案

### 4.1 架构

```
  TileLang DSL (Python)
  @T.prim_func
  def add_kernel(...):
      T.Parallel(...)
      T.copy(...)
       │
       ▼ T.compile(target="cuda" / "metal" / "rocm")
  ┌─────────────────────────────────┐
  │  TileLang Compiler               │
  │  → TILE IR → TensorIR → Target  │
  │  → Backend: triton/cute/asm/...  │
  │  → Multi-target: cuda/rocm/      │
  │    metal/vulkan/ascend/...       │
  └───────────────┬─────────────────┘
                  │ per-device binary
  ┌───────────────▼─────────────────┐
  │  TileLangKernelFactory           │
  │  (per device)                    │
  │  registered with DeviceType key  │
  └───────────────┬─────────────────┘
                  │
  ┌───────────────┼───────────────────────────────┐
  ▼               ▼                               ▼
  CUDA backend    Metal backend            ROCm backend
  (Triton/PTX)    (Metal Shader Language)  (HIP/CDNA)
```

### 4.2 跨后端内置算子示例

```python
# needle/dsl/builtin/attention.py

import tilelang as T
from needle.dsl.registry import register_tilelang_op

@register_tilelang_op(
    "flash_attention",
    device_types=["cuda", "rocm", "metal"],
    dtypes=["float16", "bfloat16"]
)
@T.prim_func
def flash_attn_fwd(
    Q: T.Buffer((batch_size, num_heads, seq_len, d_head), "float16"),
    K: T.Buffer((batch_size, num_heads, seq_len, d_head), "float16"),
    V: T.Buffer((batch_size, num_heads, seq_len, d_head), "float16"),
    O: T.Buffer((batch_size, num_heads, seq_len, d_head), "float16"),
):
    """
    Flash Attention forward.
    一次编写，自动在 CUDA / ROCm / Metal 三个后端注册。
    TileLang 编译器自动处理 tiling、shared memory staging、layout。
    """
    with T.Kernel(batch_size * num_heads, T.ceildiv(seq_len, BLOCK_M)) as bx:
        Q_shared = T.alloc_shared((BLOCK_M, d_head), "float16")
        K_shared = T.alloc_shared((BLOCK_N, d_head), "float16")
        V_shared = T.alloc_shared((BLOCK_N, d_head), "float16")
        T.copy(Q[bx, :, :], Q_shared)
        T.copy(K[bx, :, :], K_shared)
        T.copy(V[bx, :, :], V_shared)
        T.reduce(...)
```

### 4.3 Expert 模式（性能极致优化）

```python
@register_tilelang_op(
    "matmul_tma_blackwell",
    device_types=["cuda"],
    dtypes=["nvfp4"]
)
@T.prim_func
def matmul_tma_blackwell(
    A: T.Buffer((M, K), "nvfp4"),
    B: T.Buffer((K, N), "nvfp4"),
    C: T.Buffer((M, N), "float16"),
):
    """Blackwell TMA + WGMMA + NVFP4 的极致性能实现."""
    T.tma_copy(...)     # Blackwell Tensor Memory Accelerator
    T.wgmma(...)        # warp group matrix multiply-accumulate
    T.pipeline(...)     # 异步流水线
```

---

## 五、核心接口设计（C++ 侧）

### 5.1 统一 Kernel 二进制表示

```cpp
// src/fineflow/core/kernels/dsl/kernel_binary.h

namespace fineflow::dsl {

enum class Source { kTriton, kTileLang, kCustom };

struct KernelBinary {
  Source source;
  DeviceType target_device;
  std::vector<uint8_t> code;       // cubin / metallib / hip binary
  std::string entry_point;         // kernel 函数名
  std::string ptx;                 // 可选 PTX（用于 JIT 重编译）
  uint32_t min_compute_capability; // 最小 SM 版本
  uint32_t max_compute_capability; // 最大 SM 版本
};

struct LaunchGrid {
  uint32_t grid_x, grid_y, grid_z;
  uint32_t block_x, block_y, block_z;
  uint32_t shared_memory_bytes;
};

// 设备级 launcher 接口
class DeviceLauncher {
public:
  virtual ~DeviceLauncher() = default;
  virtual Ret<void> launch(const KernelBinary& binary,
                           const LaunchGrid& grid,
                           std::span<void*> args,
                           void* stream) = 0;
};

// CUDA launcher 实现
class CudaDeviceLauncher final : public DeviceLauncher {
public:
  Ret<void> launch(const KernelBinary& binary,
                   const LaunchGrid& grid,
                   std::span<void*> args,
                   void* stream) override;
};

// Metal launcher 实现
class MetalDeviceLauncher final : public DeviceLauncher {
public:
  Ret<void> launch(const KernelBinary& binary,
                   const LaunchGrid& grid,
                   std::span<void*> args,
                   void* stream) override;
};

}  // namespace fineflow::dsl
```

### 5.2 Kernel Registry 扩展

```cpp
// src/fineflow/core/kernels/dsl/dsl_registry.h

namespace fineflow::dsl {

// 注册 DSL kernel 到现有 registry
struct DSLKernelRegistry {
  // 注册到 RuntimeKernelFactoryRegistryMgr
  static void Register(
      const std::string& kernel_name,
      DeviceType device,
      KernelBinary binary,
      LaunchGrid grid_template,
      std::function<std::vector<void*>(KernelComputeContext&)> arg_extractor);

  // 批量注册（一个 TileLang 源码产生的多后端二进制）
  static void RegisterMultiDevice(
      const std::string& kernel_name,
      std::unordered_map<DeviceType, KernelBinary> device_binaries,
      LaunchGrid grid_template,
      std::function<std::vector<void*>(KernelComputeContext&)> arg_extractor);
};

}  // namespace fineflow::dsl
```

### 5.3 Python-C++ 绑定

```cpp
// src/fineflow/api/python/dsl_bridge.cpp
// pybind11 binding for DSL kernel registration

PYBIND11_MODULE(_needle_dsl_bridge, m) {
  m.def("register_dsl_kernel",
    [](const std::string& name,
       const std::string& device,
       py::bytes binary,
       py::dict meta) {
      dsl::DSLKernelRegistry::Register(name, device, ...);
    });

  m.def("register_functor",
    [](const std::string& name, py::function func) {
      // 注册 functor，使 Python 层可通过 ndl.ops.xxx() 调用
    });
}
```

---

## 六、用户自定义算子完整流程

### 6.1 场景一：Triton 手写 CUDA kernel

```python
# my_custom_kernels.py
import triton.language as tl
from needle.dsl import register_triton_op

@register_triton_op("swiglu", device_types=["cuda"],
                     dtypes=["float16", "bfloat16"])
@triton.jit
def swiglu_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr = 1024):
    """SiLU-gated linear unit."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, out, mask=mask)

# ---- 使用 ----
import needle as ndl

x = ndl.Tensor(..., device=ndl.cuda())
y = ndl.ops.swiglu(x)  # 自动 dispatch 到 Triton kernel
```

### 6.2 场景二：TileLang 跨后端算子

```python
# my_cross_backend_op.py
import tilelang as T
from needle.dsl import register_tilelang_op

@register_tilelang_op("rms_norm",
    device_types=["cuda", "rocm", "metal"],
    dtypes=["float32", "float16", "bfloat16"])
@T.prim_func
def rms_norm_fwd(
    X: T.Buffer((batch, seq, hidden), "float32"),
    W: T.Buffer((hidden,), "float32"),
    Y: T.Buffer((batch, seq, hidden), "float32"),
):
    """RMS Normalization — 一次编写，三后端可用."""
    with T.Kernel(batch * seq) as bx:
        ...
```

---

## 七、构建系统集成

### 7.1 CMake 集成

```cmake
# cmake/FindTriton.cmake
# cmake/FindTileLang.cmake

# ---- AOT 编译内置 DSL kernel ----
option(NEEDLE_AOT_DSL_KERNELS "AOT compile builtin DSL kernels" ON)

if(NEEDLE_AOT_DSL_KERNELS AND TileLang_FOUND)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m needle.dsl.compile_builtins
      --targets cuda,rocm,metal
      --output ${CMAKE_BINARY_DIR}/builtin_kernels/
      --format native
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE dsl_compile_result
  )
  if(dsl_compile_result EQUAL 0)
    # 嵌入编译产物
    file(GLOB builtin_kernel_bins ${CMAKE_BINARY_DIR}/builtin_kernels/*.bin)
    foreach(bin ${builtin_kernel_bins})
      get_filename_component(kernel_name ${bin} NAME_WE)
      file(READ ${bin} kernel_data HEX)
    endforeach()
  endif()
endif()

# ---- JIT 模式：链接 runtime adapter ----
target_sources(FineflowCore PRIVATE
  src/fineflow/core/kernels/dsl/dsl_kernel.cppm
  src/fineflow/core/kernels/dsl/cuda_launcher.cppm
  src/fineflow/core/kernels/dsl/metal_launcher.cppm   # if APPLE
)
```

### 7.2 Python 工具链

```bash
# 编译内置 DSL kernel
needle dsl compile --targets cuda,rocm,metal --output build/kernels/

# 将 JIT 缓存中的 kernel 提升为 AOT
needle dsl freeze my_custom_fusion --from-cache

# 列出所有已注册的 DSL kernel
needle dsl list

# 性能对比：DSL kernel vs 手写 C++ kernel
needle dsl bench flash_attention --compare-with=cpp
```

---

## 八、实现路线图

| 阶段 | 内容 | 交付物 |
|------|------|--------|
| **Phase 1** | Triton runtime adapter MVP | `DSLOpKernel` + `CudaDeviceLauncher`，手动注册一个 Triton kernel 端到端跑通 |
| **Phase 2** | `@register_triton_op` 完整流程 | 装饰器 + AOT 编译 + C++ 注册自动化 + functor 自动生成 |
| **Phase 3** | TileLang 集成 | `@register_tilelang_op` 装饰器，一个 kernel 同时注册到 CUDA+Metal 双后端 |
| **Phase 4** | 内置算子 DSL 迁移 | Flash Attention、MatMul、Conv 等高频算子用 TileLang 重写 |
| **Phase 5** | 用户自定义算子完整体验 | 文档、CLI 工具（`needle dsl`）、模板仓库、性能 profiling + tuning guide |
| **Phase 6** | 高级特性 | Z3 自动推理集成、自动调优（对标 tinygrad BEAM + TileLang Auto Schedule） |

---

## 九、关键设计决策

### 9.1 Triton vs TileLang 定位

建议 **Triton 作为 CUDA 入门/快速原型选项，TileLang 作为多后端内置算子的主力 DSL**：

- TileLang 的多后端可移植性与 needle 的多后端目标完全对齐
- DeepSeek V3.2/V4 已将 TileLang 用于算子快速原型，微软 BitNet/BitBLAS 基于 TileLang 开发，生态验证充分
- Triton 生态更成熟、文档更完善，适合作为入门选项
- 两者通过统一的 `DSLOpKernel` adapter 接入，底层互不排斥

### 9.2 AOT vs JIT 策略

都需要。AOT 确保内置算子的性能和确定性；JIT 保留研究探索灵活性。

关键是需要**统一的缓存层**，使 JIT 编译产物可以提升为 AOT（`needle dsl freeze`）。缓存路径：`~/.needle/kernel_cache/{source_hash}/{device}/`。

### 9.3 与 autograd 的关系

DSL kernel 只负责 forward compute。backward 有两种方式：

1. **手写 backward kernel**：用户同时注册 `my_op` 和 `my_op_backward` 两个 kernel
2. **自动 backward 生成**（远期）：在 Python autograd 层做 AOTAutograd（functionalize + trace → backward graph），自动拆解为已有 kernel

### 9.4 DSL kernel 的性能保证

- 编译期验证：`@register_*_op` 装饰器自动生成与 PyTorch 参考实现的数值对比测试
- 运行时 fallback：DSL kernel 出错时可自动 fallback 到手写 C++ kernel（若存在同名注册）
- 性能回归检测：`needle dsl bench` 命令集成到 CI
