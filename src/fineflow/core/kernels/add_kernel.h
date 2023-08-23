#include "fineflow/core/op_kernel.h"
#include "fineflow/core/op_kernel_factory.h"
namespace fineflow {

template <class T>
void EwiseAdd(const Tensor& a, const Tensor& b, Tensor* out);

class AddKernel : public OpKernel {
public:
  FF_DISALLOW_COPY_AND_MOVE(AddKernel);
  AddKernel() = default;
};

class AddKernelFactory final : public OpKernelFactory<AddKernelFactory, AddKernel> {
public:
  std::unique_ptr<AddKernel> create(DataType dtype);
};
}  // namespace fineflow
