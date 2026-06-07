module;
#include "fineflow/core/common/result.h"
#include "fineflow/core/common/util.h"
#include "kernel_factor.h"

export module fineflow.core.op_kernel.cpu.add_kernel;
import fineflow.core.op_kernel;
import fineflow.core.op_kernel_factory;
import fineflow.core.blob_tensor;
import fineflow.core.common.error;
import fineflow.core.common.data_type_proto;
import fineflow.core.common.device_type_proto;
import fineflow.core.common.fmt;
import fineflow.core.common.registry_manager;
import fineflow.core.executor.parallel_dispatch;
import std;
import std.compat;

export namespace fineflow {
DECL_KERNEL(Add);

template <class T>
class AddKernelImpl final : public AddKernel {
  void compute(KernelComputeContext& ctx) const override {
    auto in0 = *ctx.fetchTensor("in", 0);
    auto in1 = *ctx.fetchTensor("in", 1);
    auto out = *ctx.fetchTensor("out", 0);
    auto size = out.elementCount();
    T* out_ptr = out.castPtrMut<T>();
    const T* a_ptr = in0.castPtr<T>();
    const T* b_ptr = in1.castPtr<T>();
    ParallelDispatch(size, 64, [=](int gid) {
        out_ptr[gid] = a_ptr[gid] + b_ptr[gid];
    });
  }
};

IMPL_KERNEL_FACTORY(Add)
}  // namespace fineflow

namespace fineflow {
namespace {
REGISTER_KERNEL_FACTORY(Add, DeviceType::kCPU);
}  // namespace

}  // namespace fineflow
