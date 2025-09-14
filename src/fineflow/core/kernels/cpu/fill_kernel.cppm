module;
#include "fineflow/core/common/result.h"
#include "fineflow/core/common/util.h"
#include "kernel_factor.h"

export module fineflow.core.op_kernel.cpu.fill_kernel;
import fineflow.core.op_kernel;
import fineflow.core.op_kernel_factory;
import fineflow.core.blob_tensor;
import fineflow.core.common.error;
import fineflow.core.common.data_type_proto;
import fineflow.core.common.device_type_proto;
import fineflow.core.common.fmt;
import fineflow.core.common.registry_manager;
import std;
import std.compat;

export namespace fineflow {

class FillKernel : public OpKernel {
public:
  FF_DISALLOW_COPY_AND_MOVE(FillKernel);
  FillKernel() = default;
};

class FillKernelFactory final : public OpKernelFactory {
public:
  Ret<std::unique_ptr<OpKernel>> create(DataType dtype);
};
/**
 * @brief Fill buffer.
 *
 * @tparam T Kernel type.
 * @param scalar scalar
 * @param dst dst
 */
template <class T>
void Fill(const BlobTensorView& scalar, BlobTensorView& dst) {
  auto size = dst.bufferSize() / sizeof(T);
  T* out_ptr = dst.castPtrMut<T>();
  const T s = *scalar.castPtr<T>();
  for (size_t i = 0; i < size; i++) {
    out_ptr[i] = s;
  }
}

template <class T>
class FillKernelImpl final : public FillKernel {
  void compute(KernelComputeContext& ctx) const override {
    auto scalar = *ctx.fetchTensor("scalar", 0);
    auto dst = *ctx.fetchTensor("dst", 0);
    Fill<T>(scalar, dst);
  }
};

template <typename T>
std::unique_ptr<FillKernel> NewFill() {
  return std::make_unique<FillKernelImpl<T>>();
}

Ret<std::unique_ptr<OpKernel>> FillKernelFactory::create(DataType dtype) {
  static const std::map<DataType, std::function<std::unique_ptr<FillKernel>()>> new_add_handle{
      MAKE_NEW_FACTORY(NewFill)};

  auto kernel = NewKernalFromHandlers(new_add_handle, dtype);
  CHECK_OR_RETURN(kernel) << "FillKernel for type: " << std::to_string(dtype) << " has not implemented.";
  return kernel;
};
}  // namespace fineflow
namespace fineflow {
namespace {
REGISTER_KERNEL_FACTORY(FillKernel, DeviceType::kCPU, FillKernelFactory);
}  // namespace
}  // namespace fineflow
