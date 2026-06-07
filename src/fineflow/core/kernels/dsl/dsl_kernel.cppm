module;
#include "fineflow/core/common/result.h"
#include "fineflow/core/common/util.h"

export module fineflow.core.kernels.dsl.dsl_kernel;

import std;
import fineflow.core.op_kernel;
import fineflow.core.op_kernel_factory;
import fineflow.core.kernels.dsl.device_launcher;
import fineflow.core.common.device_type_proto;
import fineflow.core.common.data_type_proto;
import fineflow.core.common.error;

export namespace fineflow::dsl {

class DSLOpKernel final : public OpKernel {
public:
  DSLOpKernel(DSLKernelMeta meta, std::unique_ptr<DeviceLauncher> launcher)
      : meta_(std::move(meta)), launcher_(std::move(launcher)) {}

  void compute(KernelComputeContext& ctx) const override {
    (void)launcher_->launch(meta_, ctx);
  }

private:
  DSLKernelMeta meta_;
  std::unique_ptr<DeviceLauncher> launcher_;
};

class DSLOpKernelFactory final : public OpKernelFactory {
public:
  explicit DSLOpKernelFactory(DSLKernelMeta meta) : meta_(std::move(meta)) {}

  Ret<std::unique_ptr<OpKernel>> create(DataType /*dtype*/) override {
    std::unique_ptr<DeviceLauncher> launcher;
    // Use native launcher if compiled library path is available
    if (meta_.target_device == DeviceType::kCPU && !meta_.native_lib_path.empty()) {
      launcher = std::unique_ptr<DeviceLauncher>(
          new NativeCpuDeviceLauncher(meta_.native_lib_path));
    } else {
      TRY_ASSIGN(launcher, DeviceLauncher::New(meta_.target_device));
    }
    return std::unique_ptr<OpKernel>(new DSLOpKernel(meta_, std::move(launcher)));
  }

private:
  DSLKernelMeta meta_;
};

}  // namespace fineflow::dsl
