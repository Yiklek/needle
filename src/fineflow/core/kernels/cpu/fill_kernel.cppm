module;
#include "fineflow/core/op_kernel.h"
#include "fineflow/core/op_kernel_factory.h"
export module cpu_fill_kernel;
export {
#include "fineflow/core/kernels/fill_kernel.h"
}
namespace fineflow {

namespace {
REGISTER_KERNEL(DeviceType::kCPU, FillKernelFactory);
/**
 * @brief Fill buffer.
 *
 * @tparam T Kernel type.
 * @param scalar scalar
 * @param dst dst
 */
template <class T>
void Fill(const BlobTensorView& scalar, BlobTensorView* dst) {
  auto size = dst->bufferSize() / sizeof(T);
  T* out_ptr = dst->castPtrMut<T>();
  const T s = *scalar.castPtr<T>();
  for (size_t i = 0; i < size; i++) {
    out_ptr[i] = s;
  }
}

template <class T>
class FillKernelImpl final : public FillKernel {
  void compute(KernelComputeContext* ctx) const override {
    auto scalar = *ctx->fetchTensor("scalar", 0);
    auto dst = *ctx->fetchTensor("dst", 0);
    Fill<T>(scalar, &dst);
  }
};

template <typename T>
std::unique_ptr<FillKernel> NewFill() {
  return std::make_unique<FillKernelImpl<T>>();
}
}  // namespace

Ret<std::unique_ptr<FillKernel>> FillKernelFactory::create(DataType dtype) {
  static const std::map<DataType, std::function<std::unique_ptr<FillKernel>()>> new_add_handle{

#define MAKE_NEW_COMPACT_ENTRY(type_cpp, type_proto) {type_proto, NewFill<type_cpp>},
#define FOR_MAKE_NEW_COMPACT_ENTRY(i, data, elem) FF_PP_FORWARD(MAKE_NEW_COMPACT_ENTRY, BOOST_PP_TUPLE_ENUM(elem))
      BOOST_PP_SEQ_FOR_EACH(FOR_MAKE_NEW_COMPACT_ENTRY, _, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)
#undef FOR_MAKE_NEW_COMPACT_ENTRY
#undef MAKE_NEW_COMPACT_ENTRY

  };

  auto kernel = NewKernalFromHandlers(new_add_handle, dtype);
  CHECK_OR_RETURN(kernel) << "FillKernel for type: " << fmt::to_string(dtype) << " has not implemented.";
  return kernel;
};
}  // namespace fineflow
