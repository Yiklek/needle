#ifndef FINEFLOW_CORE_KERNELS_ASSIGN_KERNEL_H_
#define FINEFLOW_CORE_KERNELS_ASSIGN_KERNEL_H_
#include "fineflow/core/op_kernel.h"
#include "fineflow/core/op_kernel_factory.h"

namespace fineflow {

class AssignKernel : public OpKernel {
public:
  FF_DISALLOW_COPY_AND_MOVE(AssignKernel);
  AssignKernel() = default;
};

class AssignKernelFactory final : public OpKernelFactory<AssignKernelFactory, AssignKernel> {
public:
  static Ret<std::unique_ptr<AssignKernel>> create(DataType dtype);
};
}  // namespace fineflow
#endif  // FINEFLOW_CORE_KERNELS_ADD_KERNEL_H_
