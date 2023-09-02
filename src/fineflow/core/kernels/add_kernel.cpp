// #include "fineflow/core/common/auto_register.hpp"
#include "fineflow/core/kernels/add_kernel.h"

#include "fineflow/core/common/device_type.pb.h"
#include "fineflow/core/common/registry_manager.hpp"
#include "fineflow/core/op_kernel_factory.h"
namespace fineflow {

template <class T>
void EwiseAdd(const Tensor& a, const Tensor& b, Tensor* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  auto size = out->elementCount();
  T* out_ptr = out->castPtrMut<T>();
  const T* a_ptr = a.castPtr<T>();
  const T* b_ptr = b.castPtr<T>();
  for (size_t i = 0; i < size; i++) {
    out_ptr[i] = a_ptr[i] + b_ptr[i];
  }
}

template <class T>
class AddKernelImpl final : public AddKernel {
  void compute(KernelComputeContext* ctx) const override {
    auto in0 = ctx->fetchTensor("in", 0).value();
    auto in1 = ctx->fetchTensor("in", 1).value();
    auto out = ctx->fetchTensor("out", 0).value();
    EwiseAdd<T>(*in0, *in1, out.get());
  }
};

template <typename T>
std::unique_ptr<AddKernel> NewAdd() {
  return std::unique_ptr<AddKernel>(new AddKernelImpl<T>());
}

std::unique_ptr<AddKernel> AddKernelFactory::create(DataType dtype) {
  static const std::map<DataType, std::function<std::unique_ptr<AddKernel>()>> new_add_handle{
      {DataType::kFloat, NewAdd<float>}};

  return NewKernalFromHandlers(new_add_handle, dtype);
};
namespace {
// REGISTER_CLASS(DeviceType, DeviceType::kCPU, AddKernelFactory, AddKernelFactory);
REGISTER_KERNEL(DeviceType::kCPU, AddKernelFactory);
// REGISTER_KEY_WITH_CLASS(DeviceType, std::function<AddKernelFactory*()>, DeviceType::kCPU).setValue([] {
//   return new AddKernelFactory();
// });
}  // namespace
}  // namespace fineflow
