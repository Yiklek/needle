#include "fineflow/core/functional/basic_functor.h"

#include "fineflow/core/common/result.hpp"
#include "fineflow/core/cpu/cpu_tensor.h"
#include "fineflow/core/kernels/add_kernel.h"
#include "fineflow/core/kernels/compact_kernel.h"
#include "fineflow/core/tensor_util.h"
namespace fineflow {
namespace {
REGISTER_KEY(std::function<AddFunctorType>, std::string("add")).setValue(std::function<AddFunctorType>(AddFunctor()));
REGISTER_KEY(std::function<CompactFunctorType>, std::string("compact"))
    .setValue(std::function<CompactFunctorType>(CompactFunctor()));
}  // namespace
// class AddFunctor {
// public:
//   AddFunctor() {
//     //   op_.resize(kMaxInputCount /*the maximum number of inputs*/);
//     //   for (int n = 1; n < op_.size(); ++n) {
//     //     op_[n] = CHECK_JUST(one::OpBuilder("add_n").Input("in", n + 1).Output("out").Build());
//     //   }
//   }
//   BlobTensorPtr operator()(const BlobTensorPtr a, const BlobTensorPtr b) {
//     if (a->dtype() != b->dtype()) {
//       auto e = "a and b must be same dtype.";
//       SPDLOG_ERROR(e);
//       throw std::invalid_argument(e);
//     }
//     auto tc = std::shared_ptr<BlobTensor>(new CpuTensor(a->dtype(), a->shape()));
//     KernelComputeContext ctx;
//     ctx.arg2tensor_.insert({{"in", 0}, std::move(a)});
//     ctx.arg2tensor_.insert({{"in", 1}, std::move(b)});
//     ctx.arg2tensor_.insert({{"out", 0}, std::move(tc)});
//     auto &f = **KernelRegistryMgr<AddKernelFactory>::Get().GetValue(DeviceType::kCPU);
//     f->create(a->dtype())->compute(&ctx);
//     return tc;
//   }
//
//   // private:
//   //   std::vector<std::shared_ptr<OpExpr>> op_;
// };
template <class T>
inline Ret<void> Call(KernelComputeContext& ctx, DataType kernel_dtype) {
  TRY_ASSIGN(auto f, KernelRegistryMgr<T>::Get().GetValue(DeviceType::kCPU));
  TRY_ASSIGN(auto kernel, (*f)->create(kernel_dtype));
  kernel->compute(&ctx);
  return {};
}

Ret<BlobTensorPtr> AddFunctor::operator()(const BlobTensorPtr& a, const BlobTensorPtr& b) {
  if (a->dtype() != b->dtype()) {
    auto e = fmt::format("Tensor a({}) and Tensor b({}) must be same dtype.", a->dtype(), b->dtype());
    SPDLOG_ERROR(e);
    throw std::invalid_argument(e);
  }
  auto tc = std::shared_ptr<BlobTensor>(new CpuTensor(a->dtype(), a->shape()));
  KernelComputeContext ctx;
  ctx.arg2tensor_.insert({{"in", 0}, std::move(a)});
  ctx.arg2tensor_.insert({{"in", 1}, std::move(b)});
  ctx.arg2tensor_.insert({{"out", 0}, tc});
  TRY(Call<AddKernelFactory>(ctx, a->dtype()));
  // TRY_ASSIGN(auto f, KernelRegistryMgr<AddKernelFactory>::Get().GetValue(DeviceType::kCPU));
  // TRY_ASSIGN(auto kernel, (*f)->create(a->dtype()));
  // kernel->compute(&ctx);
  return tc;
}

Ret<BlobTensorPtr> CompactFunctor::operator()(const BlobTensorPtr& a) {
  auto r = DeriveEmptyTensorLike(a);
  CHECK_OR_RETURN(r);
  KernelComputeContext ctx;
  ctx.arg2tensor_.insert({{"in", 0}, std::move(a)});
  ctx.arg2tensor_.insert({{"out", 0}, r});
  TRY(Call<CompactKernelFactory>(ctx, a->dtype()));
  // TRY_ASSIGN(auto f, KernelRegistryMgr<CompactKernelFactory>::Get().GetValue(DeviceType::kCPU));
  // TRY_ASSIGN(auto kernel, (*f)->create(a->dtype()));
  // kernel->compute(&ctx);
  return r;
}

}  // namespace fineflow
