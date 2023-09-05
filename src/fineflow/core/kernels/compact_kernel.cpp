#include "fineflow/core/kernels/compact_kernel.h"

#include "fineflow/core/common/device_type.pb.h"
#include "fineflow/core/common/registry_manager.hpp"
#include "fineflow/core/op_kernel_factory.h"
namespace fineflow {

template <class T>
void Compact(const BlobTensor& a, BlobTensor* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  auto size = out->elementCount();
  T* out_ptr = out->castPtrMut<T>();
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
  void compute(KernelComputeContext* ctx) const override {
    auto in0 = ctx->fetchTensor("in", 0).value();
    auto out = ctx->fetchTensor("out", 0).value();
    Compact<T>(*in0, out.get());
  }
};

template <typename T>
std::unique_ptr<CompactKernel> NewCompact() {
  return std::unique_ptr<CompactKernel>(new CompactKernelImpl<T>());
}

Ret<std::unique_ptr<CompactKernel>> CompactKernelFactory::create(DataType dtype) {
  static const std::map<DataType, std::function<std::unique_ptr<CompactKernel>()>> new_add_handle{

#define MAKE_NEW_ADD_ENTRY(type_cpp, type_proto) {type_proto, NewCompact<type_cpp>},
// for i in [0, CPU_PRIMITIVE_NATIVE_TYPE_TUPLE)
// #define BOOST_PP_LOCAL_LIMITS (0, BOOST_PP_TUPLE_SIZE(CPU_PRIMITIVE_NATIVE_TYPE_TUPLE) - 1)
// #define BOOST_PP_LOCAL_MACRO(i) \
//   FF_PP_FORWARD(MAKE_NEW_ADD_ENTRY, BOOST_PP_TUPLE_ENUM(BOOST_PP_TUPLE_ELEM(i, CPU_PRIMITIVE_NATIVE_TYPE_TUPLE)))
// #include BOOST_PP_LOCAL_ITERATE()
#define FOR_MAKE_NEW_ADD_ENTRY(i, data, elem) FF_PP_FORWARD(MAKE_NEW_ADD_ENTRY, BOOST_PP_TUPLE_ENUM(elem))
      BOOST_PP_SEQ_FOR_EACH(FOR_MAKE_NEW_ADD_ENTRY, _, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)
#undef FOR_MAKE_NEW_ADD_ENTRY
#undef MAKE_NEW_ADD_ENTRY

  };

  auto kernel = NewKernalFromHandlers(new_add_handle, dtype);
  CHECK_OR_RETURN(kernel) << "AddKernel for type: " << fmt::to_string(dtype) << " has not implemented.";
  return kernel;
};
namespace {
REGISTER_KERNEL(DeviceType::kCPU, CompactKernelFactory);
}  // namespace
}  // namespace fineflow
