#ifndef FINEFLOW_CORE_KERNELS_COMPACT_KERNEL_H_
#define FINEFLOW_CORE_KERNELS_COMPACT_KERNEL_H_

#include "fineflow/core/op_kernel.h"
#include "fineflow/core/op_kernel_factory.h"

namespace fineflow {

class CompactKernel : public OpKernel {
public:
  FF_DISALLOW_COPY_AND_MOVE(CompactKernel);
  CompactKernel() = default;
};

class CompactKernelFactory final : public OpKernelFactory<CompactKernelFactory, CompactKernel> {
public:
  static Ret<std::unique_ptr<CompactKernel>> create(DataType dtype);
};
}  // namespace fineflow
#endif  // FINEFLOW_CORE_KERNELS_COMPACT_KERNEL_H_
