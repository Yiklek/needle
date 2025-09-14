module;
#include "fineflow/core/common/result.h"
#include "fineflow/core/common/util.h"
#include "kernel_factor.h"

export module fineflow.core.op_kernel.cpu.add_kernel;
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
class AddKernelFactory;
class AddKernel : public OpKernel {
public:
  FF_DISALLOW_COPY_AND_MOVE(AddKernel);
  AddKernel() = default;
};

class AddKernelFactory final : public OpKernelFactory<AddKernelFactory, AddKernel> {
public:
  static Ret<std::unique_ptr<AddKernel>> create(DataType dtype);
};
template <class T>
void EwiseAdd(const BlobTensorView& a, const BlobTensorView& b, BlobTensorView* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  auto size = out->elementCount();
  T* out_ptr = out->castPtrMut<T>();
  const T* a_ptr = a.castPtr<T>();
  const T* b_ptr = b.castPtr<T>();

  // #pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    out_ptr[i] = a_ptr[i] + b_ptr[i];
  }
}
template <class T>
class AddKernelImpl final : public AddKernel {
  void compute(KernelComputeContext* ctx) const override {
    auto in0 = *ctx->fetchTensor("in", 0);
    auto in1 = *ctx->fetchTensor("in", 1);
    auto out = *ctx->fetchTensor("out", 0);
    EwiseAdd<T>(in0, in1, &out);
  }
};
template <typename T>
std::unique_ptr<AddKernel> NewAdd() {
  return std::make_unique<AddKernelImpl<T>>();
}
Ret<std::unique_ptr<AddKernel>> AddKernelFactory::create(DataType dtype) {
  static const std::map<DataType, std::function<std::unique_ptr<AddKernel>()>> new_add_handle{MAKE_NEW_FACTORY(NewAdd)};

  auto kernel = NewKernalFromHandlers(new_add_handle, dtype);
  CHECK_OR_RETURN(kernel) << "AddKernel for type: " << std::to_string(dtype) << " has not implemented.";
  return kernel;
};
}  // namespace fineflow

namespace fineflow {
namespace {
REGISTER_KERNEL_FACTORY(DeviceType::kCPU, AddKernelFactory);
}  // namespace

}  // namespace fineflow
