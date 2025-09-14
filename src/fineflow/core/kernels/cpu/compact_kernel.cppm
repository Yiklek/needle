module;
#include "fineflow/core/common/result.h"
#include "fineflow/core/common/util.h"
#include "kernel_factor.h"

export module fineflow.core.op_kernel.cpu.compact_kernel;
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

class CompactKernel : public OpKernel {
public:
  FF_DISALLOW_COPY_AND_MOVE(CompactKernel);
  CompactKernel() = default;
};

class CompactKernelFactory final : public OpKernelFactory {
public:
  Ret<std::unique_ptr<OpKernel>> create(DataType dtype);
};
template <class T>
void Compact(const BlobTensorView& a, BlobTensorView& out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  auto size = out.elementCount();
  T* out_ptr = out.castPtrMut<T>();
  const T* a_ptr = a.castPtr<T>();
  // for (size_t i = 0; i < size; i++) {
  //   out_ptr[i] = a_ptr[i] + ;
  // }
  const auto& shape = a.shape();
  size_t dim = shape.size();
  const auto& strides = a.stride();
  auto offset = a.offset();
  // // NOTE uint32_t has changed to int32_t
  std::vector<int32_t> pos(dim, 0);
  for (int32_t i = 0; i < size; i++) {
    int32_t idx = 0;
    for (int32_t j = 0; j < dim; j++) idx += strides[dim - 1 - j] * pos[j];
    out_ptr[i] = a_ptr[idx + offset];
    pos[0] += 1;
    // carry
    for (int32_t j = 0; j < dim; j++) {
      if (pos[j] == shape[dim - 1 - j]) {
        pos[j] = 0;
        if (j != dim - 1) pos[j + 1] += 1;
      }
    }
  }
}

template <class T>
class CompactKernelImpl final : public CompactKernel {
  void compute(KernelComputeContext& ctx) const override {
    auto in0 = *ctx.fetchTensor("in", 0);
    auto out = *ctx.fetchTensor("out", 0);
    Compact<T>(in0, out);
  }
};

template <typename T>
std::unique_ptr<CompactKernel> NewCompact() {
  return std::make_unique<CompactKernelImpl<T>>();
}

Ret<std::unique_ptr<OpKernel>> CompactKernelFactory::create(DataType dtype) {
  static const std::map<DataType, std::function<std::unique_ptr<CompactKernel>()>> new_add_handle{
      MAKE_NEW_FACTORY(NewCompact)};

  auto kernel = NewKernalFromHandlers(new_add_handle, dtype);
  CHECK_OR_RETURN(kernel) << "AddKernel for type: " << std::to_string(dtype) << " has not implemented.";
  return kernel;
};
}  // namespace fineflow
namespace fineflow {
namespace {
REGISTER_KERNEL_FACTORY(CompactKernel, DeviceType::kCPU, CompactKernelFactory);
}  // namespace

}  // namespace fineflow
