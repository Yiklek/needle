module;
#include "fineflow/core/common/result.h"
#include "fineflow/core/common/util.h"
#include <cstdint>

export module fineflow.core.kernels.dsl.device_launcher;

import std;
import fineflow.core.op_kernel;
import fineflow.core.common.device_type_proto;
import fineflow.core.common.data_type_proto;
import fineflow.core.common.error;
import fineflow.core.common.log;

export namespace fineflow::dsl {

enum class Source { kTriton, kTileLang, kCustom };

struct DSLKernelMeta {
  std::string name;
  Source source_type = Source::kTileLang;
  DeviceType target_device = DeviceType::kInvalidDevice;

  // CPU: function pointer (no binary needed)
  std::function<void(KernelComputeContext&)> cpu_compute;

  // Metal: .metal source code (JIT compiled to metallib at first call)
  std::string metal_source;
  std::string entry_point;

  // Operator attributes (compile-time defaults)
  AttrMap attrs_schema;

  // GPU: binary blob
  std::vector<uint8_t> binary;
};

class DeviceLauncher {
public:
  FF_DISALLOW_COPY_AND_MOVE(DeviceLauncher);
  virtual ~DeviceLauncher() = default;

  virtual Ret<void> launch(const DSLKernelMeta& meta, KernelComputeContext& ctx) = 0;

  static Ret<std::unique_ptr<DeviceLauncher>> New(DeviceType device);

protected:
  DeviceLauncher() = default;
};

// CPU device launcher — executes meta.cpu_compute directly
class CpuDeviceLauncher final : public DeviceLauncher {
public:
  CpuDeviceLauncher() = default;

  Ret<void> launch(const DSLKernelMeta& meta, KernelComputeContext& ctx) override {
    CHECK_OR_RETURN(meta.cpu_compute) << "CPU launcher requires cpu_compute function in meta";
    meta.cpu_compute(ctx);
    return {};
  }
};

// Metal device launcher — stub for macOS Metal support
class MetalDeviceLauncher final : public DeviceLauncher {
public:
  MetalDeviceLauncher() = default;
  Ret<void> launch(const DSLKernelMeta& /*meta*/, KernelComputeContext& /*ctx*/) override {
    return UNIMPLEMENTED_ERROR;
  }
};

// CUDA device launcher — stub for NVIDIA CUDA support
class CudaDeviceLauncher final : public DeviceLauncher {
public:
  CudaDeviceLauncher() = default;
  Ret<void> launch(const DSLKernelMeta& /*meta*/, KernelComputeContext& /*ctx*/) override {
    return UNIMPLEMENTED_ERROR;
  }
};

inline Ret<std::unique_ptr<DeviceLauncher>> DeviceLauncher::New(DeviceType device) {
  switch (device) {
    case DeviceType::kCPU:
      return std::unique_ptr<DeviceLauncher>(new CpuDeviceLauncher());
    case DeviceType::kMetal:
      return std::unique_ptr<DeviceLauncher>(new MetalDeviceLauncher());
    case DeviceType::kCUDA:
      return std::unique_ptr<DeviceLauncher>(new CudaDeviceLauncher());
    default:
      CHECK_OR_RETURN(false) << "Unsupported device type for DSL launcher: " << static_cast<int>(device);
      return {};
  }
}

}  // namespace fineflow::dsl
