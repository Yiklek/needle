#include "fineflow/core/common/result.hpp"
#include "fineflow/core/cpu/cpu_tensor.h"
#include "fineflow/core/functional/basic_functor.h"
#include "fineflow/core/kernels/add_kernel.h"
#include "fineflow/core/kernels/compact_kernel.h"
#include "fineflow/core/tensor_util.h"

namespace fineflow {
namespace {
REGISTER_FUNCTOR(AddFunctor, "add")
REGISTER_FUNCTOR(CompactFunctor, "compact")
}  // namespace
template <class T>
inline Ret<void> Call(KernelComputeContext& ctx, DataType kernel_dtype) {
  TRY_ASSIGN(auto f, KernelRegistryMgr<T>::Get().GetValue(DeviceType::kCPU));
  TRY_ASSIGN(auto kernel, (*f)->create(kernel_dtype));
  kernel->compute(&ctx);
  return {};
}

Ret<BlobTensorPtr> AddFunctor::operator()(const BlobTensorPtr& a, const BlobTensorPtr& b) {
  CHECK_OR_RETURN(a->dtype() == b->dtype())
      << fmt::format("Tensor a({}) and Tensor b({}) must be same dtype.", a->dtype(), b->dtype());

  auto tc = std::shared_ptr<BlobTensor>(new CpuTensor(a->dtype(), a->shape()));
  KernelComputeContext ctx;
  ctx.arg2tensor_.insert({{"in", 0}, std::move(a)});
  ctx.arg2tensor_.insert({{"in", 1}, std::move(b)});
  ctx.arg2tensor_.insert({{"out", 0}, tc});
  TRY(Call<AddKernelFactory>(ctx, a->dtype()));
  return tc;
}

Ret<BlobTensorPtr> CompactFunctor::operator()(const BlobTensorPtr& a) {
  auto r = DeriveEmptyTensorLike(a);
  CHECK_OR_RETURN(r);
  KernelComputeContext ctx;
  ctx.arg2tensor_.insert({{"in", 0}, std::move(a)});
  ctx.arg2tensor_.insert({{"out", 0}, r});
  TRY(Call<CompactKernelFactory>(ctx, a->dtype()));
  return r;
}

}  // namespace fineflow
