#ifndef FINEFLOW_CORE_KERNELS_FILL_KERNEL_H_
#define FINEFLOW_CORE_KERNELS_FILL_KERNEL_H_
#include "fineflow/core/op_kernel.h"
#include "fineflow/core/op_kernel_factory.h"

namespace fineflow {

class FillKernel : public OpKernel {
public:
  FF_DISALLOW_COPY_AND_MOVE(FillKernel);
  FillKernel() = default;
};

class FillKernelFactory final : public OpKernelFactory<FillKernelFactory, FillKernel> {
public:
  static Ret<std::unique_ptr<FillKernel>> create(DataType dtype);
};
}  // namespace fineflow
#endif  // FINEFLOW_CORE_KERNELS_ADD_KERNEL_H_
