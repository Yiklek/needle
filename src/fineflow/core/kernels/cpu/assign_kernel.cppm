module;
#include "fineflow/core/common/result.h"
#include "fineflow/core/common/util.h"
#include "kernel_factor.h"

export module fineflow.core.op_kernel.cpu.assign_kernel;
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
DECL_KERNEL(Assign);

template <class T>
class AssignKernelImpl final : public AssignKernel {
  void compute(KernelComputeContext& ctx) const override {
    auto src = *ctx.fetchTensor("src", 0);
    auto dst = *ctx.fetchTensor("dst", 0);
    auto shape = dst.shape();
    auto strides = dst.stride();
    int32_t dim = static_cast<int32_t>(shape.size());
    auto* src_ptr = src.castPtr<T>() + src.offset();
    auto* dst_ptr = dst.castPtrMut<T>() + dst.offset();
    bool scalar = src.isScalar();
    const T scalar_val = scalar ? *src_ptr : T{};

    ParallelDispatch(dst.elementCount(), 64, [=](int gid) {
        int32_t idx = 0;
        int32_t rem = gid;
        for (int j = dim - 1; j >= 0; j--) {
            int32_t coord = rem % static_cast<int32_t>(shape[j]);
            rem /= static_cast<int32_t>(shape[j]);
            idx += static_cast<int32_t>(strides[j]) * coord;
        }
        dst_ptr[idx] = scalar ? scalar_val : src_ptr[gid];
    });
  }
};

IMPL_KERNEL_FACTORY(Assign)
}  // namespace fineflow
namespace fineflow {
namespace {
REGISTER_KERNEL_FACTORY(Assign, DeviceType::kCPU);
}
}  // namespace fineflow
