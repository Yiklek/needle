module;
#include "fineflow/core/common/result.h"
#include "fineflow/core/common/util.h"
#include "kernel_factor.h"

export module fineflow.core.op_kernel.cpu.assign_kernel;
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

class AssignKernel : public OpKernel {
public:
  FF_DISALLOW_COPY_AND_MOVE(AssignKernel);
  AssignKernel() = default;
};

class AssignKernelFactory final : public OpKernelFactory<AssignKernelFactory, AssignKernel> {
public:
  static Ret<std::unique_ptr<AssignKernel>> create(DataType dtype);
};
/**
 * @brief Assign buffer.
 *
 * @tparam T Kernel type.
 * @param src Source to assign. If src is not scalar, shape must be same to dst.
 * @param dst Dest to assign.
 */
template <class T>
void Assign(const BlobTensorView& src, BlobTensorView* dst) {
  auto shape = dst->shape();
  auto strides = dst->stride();
  int32_t dim = shape.size();
  auto* src_ptr = src.castPtr<T>() + src.offset();
  auto* dst_ptr = dst->castPtrMut<T>() + dst->offset();
  auto scalar = src.isScalar();
  auto get_elem = std::function([&](size_t idx) { return src_ptr[idx]; });
  if (scalar) {
    get_elem = std::function([&](size_t) { return *src_ptr; });
  }
  // NOTE uint32_t has changed to int32_t
  std::vector<int32_t> pos(dim, 0);
  // NOTE careful with the iteration times, not `out-size`!
  for (size_t i = 0; i < dst->elementCount(); i++) {
    int32_t idx = 0;
    for (int j = 0; j < dim; j++) idx += strides[dim - 1 - j] * pos[j];
    dst_ptr[idx] = get_elem(idx);
    pos[0] += 1;
    // carry
    for (int j = 0; j < dim; j++) {
      if (pos[j] == shape[dim - 1 - j]) {
        pos[j] = 0;
        if (j != dim - 1) pos[j + 1] += 1;
      }
    }
  }
}

template <class T>
class AssignKernelImpl final : public AssignKernel {
  void compute(KernelComputeContext* ctx) const override {
    auto src = *ctx->fetchTensor("src", 0);
    auto dst = *ctx->fetchTensor("dst", 0);
    Assign<T>(src, &dst);
  }
};

template <typename T>
std::unique_ptr<AssignKernel> NewAssign() {
  return std::make_unique<AssignKernelImpl<T>>();
}

Ret<std::unique_ptr<AssignKernel>> AssignKernelFactory::create(DataType dtype) {
  static const std::map<DataType, std::function<std::unique_ptr<AssignKernel>()>> new_add_handle{
      MAKE_NEW_FACTORY(NewAssign)};

  auto kernel = NewKernalFromHandlers(new_add_handle, dtype);
  CHECK_OR_RETURN(kernel) << "AssignKernel for type: " << std::to_string(dtype) << " has not implemented.";
  return kernel;
};
}  // namespace fineflow
namespace fineflow {
namespace {
REGISTER_KERNEL_FACTORY(DeviceType::kCPU, AssignKernelFactory);
}
}  // namespace fineflow
