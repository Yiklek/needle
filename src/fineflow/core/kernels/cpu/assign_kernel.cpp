#include "fineflow/core/common/device_type.pb.h"
#include "fineflow/core/common/registry_manager.hpp"
#include "fineflow/core/kernels/assign_kernel.h"
#include "fineflow/core/op_kernel_factory.h"
namespace fineflow {

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
  return std::unique_ptr<AssignKernel>(new AssignKernelImpl<T>());
}

Ret<std::unique_ptr<AssignKernel>> AssignKernelFactory::create(DataType dtype) {
  static const std::map<DataType, std::function<std::unique_ptr<AssignKernel>()>> new_add_handle{

#define MAKE_NEW_COMPACT_ENTRY(type_cpp, type_proto) {type_proto, NewAssign<type_cpp>},
#define FOR_MAKE_NEW_COMPACT_ENTRY(i, data, elem) FF_PP_FORWARD(MAKE_NEW_COMPACT_ENTRY, BOOST_PP_TUPLE_ENUM(elem))
      BOOST_PP_SEQ_FOR_EACH(FOR_MAKE_NEW_COMPACT_ENTRY, _, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)
#undef FOR_MAKE_NEW_COMPACT_ENTRY
#undef MAKE_NEW_COMPACT_ENTRY

  };

  auto kernel = NewKernalFromHandlers(new_add_handle, dtype);
  CHECK_OR_RETURN(kernel) << "AssignKernel for type: " << fmt::to_string(dtype) << " has not implemented.";
  return kernel;
};
namespace {
REGISTER_KERNEL(DeviceType::kCPU, AssignKernelFactory);
}  // namespace
}  // namespace fineflow
