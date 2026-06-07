module;
#include "fineflow/core/common/result.h"
#include "fineflow/core/common/util.h"
#include <cstdint>
#include <dlfcn.h>

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

  // Native compiled library path (.dylib/.so) for dlopen
  std::string native_lib_path;

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

// CPU device launcher — Python bridge (development/fallback)
class CpuDeviceLauncher final : public DeviceLauncher {
public:
  CpuDeviceLauncher() = default;

  Ret<void> launch(const DSLKernelMeta& meta, KernelComputeContext& ctx) override {
    CHECK_OR_RETURN(meta.cpu_compute) << "CPU launcher requires cpu_compute function in meta";
    meta.cpu_compute(ctx);
    return {};
  }
};

// Native CPU device launcher — dlopen compiled .dylib/.so
class NativeCpuDeviceLauncher final : public DeviceLauncher {
public:
  explicit NativeCpuDeviceLauncher(std::string lib_path)
      : lib_path_(std::move(lib_path)) {}

  Ret<void> launch(const DSLKernelMeta& meta, KernelComputeContext& ctx) override {
    // Attempt native execution: dlopen the compiled library
    if (!lib_path_.empty()) {
      void* handle = dlopen(lib_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (handle != nullptr) {
        dlerror();
        std::string symbol_name = "_" + meta.entry_point;
        void* fn = dlsym(handle, symbol_name.c_str());
        if (fn != nullptr) {
          // Native kernel symbol found. TVM FFI calling convention requires
          // DLTensor argument packing — currently delegated to Python bridge.
          // Future: pack DLTensors from ctx tensors and call fn directly.
          dlclose(handle);
        } else {
          dlclose(handle);
        }
      }
    }
    // Fallback to Python bridge for execution
    if (meta.cpu_compute) {
      meta.cpu_compute(ctx);
      return {};
    }
    return UNIMPLEMENTED_ERROR;
  }

private:
  std::string lib_path_;
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
