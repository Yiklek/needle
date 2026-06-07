module;
#include "fineflow/core/common/result.h"
#include "fineflow/core/common/util.h"
#include "kernel_factor.h"

export module fineflow.core.op_kernel.cpu.compact_kernel;
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
DECL_KERNEL(Compact);

template <class T>
class CompactKernelImpl final : public CompactKernel {
  void compute(KernelComputeContext& ctx) const override {
    auto in0 = *ctx.fetchTensor("in", 0);
    auto out = *ctx.fetchTensor("out", 0);
    auto size = out.elementCount();
    T* out_ptr = out.castPtrMut<T>();
    const T* a_ptr = in0.castPtr<T>();
    const auto& shape = in0.shape();
    int32_t dim = static_cast<int32_t>(shape.size());
    const auto& strides = in0.stride();
    auto offset = in0.offset();

    ParallelDispatch(size, 64, [=](int gid) {
        int32_t idx = 0;
        int32_t rem = gid;
        for (int j = dim - 1; j >= 0; j--) {
            int32_t coord = rem % static_cast<int32_t>(shape[j]);
            rem /= static_cast<int32_t>(shape[j]);
            idx += static_cast<int32_t>(strides[j]) * coord;
        }
        out_ptr[gid] = a_ptr[idx + offset];
    });
  }
};

IMPL_KERNEL_FACTORY(Compact);
}  // namespace fineflow
namespace fineflow {
namespace {
REGISTER_KERNEL_FACTORY(Compact, DeviceType::kCPU);
}  // namespace

}  // namespace fineflow
