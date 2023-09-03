#include "fineflow/core/functional/basic_functor.h"

#include "fineflow/core/common/result.hpp"
#include "fineflow/core/cpu/cpu_tensor.h"
#include "fineflow/core/kernels/add_kernel.h"
namespace fineflow {
namespace {
REGISTER_KEY(std::function<AddFunctorType>, std::string("add")).setValue(std::function<AddFunctorType>(AddFunctor()));
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
  TRY_ASSIGN(auto f, KernelRegistryMgr<AddKernelFactory>::Get().GetValue(DeviceType::kCPU));
  TRY_ASSIGN(auto kernel, (*f)->create(a->dtype()));
  kernel->compute(&ctx);
  return tc;
}

}  // namespace fineflow
