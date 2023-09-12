#include "fineflow/core/common/preprocess.h"
#include "fineflow/core/common/result.hpp"
#include "fineflow/core/cpu/cpu_tensor.h"
#include "fineflow/core/functional/basic_functor.h"
#include "fineflow/core/kernels/add_kernel.h"
#include "fineflow/core/kernels/compact_kernel.h"
#include "fineflow/core/kernels/fill_kernel.h"
#include "fineflow/core/tensor_util.h"

namespace fineflow {

namespace {
REGISTER_FUNCTOR(AddFunctor, "add")
REGISTER_FUNCTOR(CompactFunctor, "compact")
REGISTER_FUNCTOR(FillFunctor<const BlobTensorPtr&>, "fill")
#define REGISTER_FILL_FUNCTOR(type_cpp, type_proto) REGISTER_FUNCTOR(FillFunctor<type_cpp>, "fill")
#define FOR_REGISTER_FULL_FUNCTOR(i, data, elem) REGISTER_FILL_FUNCTOR elem
BOOST_PP_SEQ_FOR_EACH(FOR_REGISTER_FULL_FUNCTOR, _, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)
#undef FOR_REGISTER_FULL_FUNCTOR
#undef REGISTER_FILL_FUNCTOR
}  // namespace

template <class T>
inline Ret<void> Call(KernelComputeContext& ctx) {
  TRY_ASSIGN(auto f, KernelRegistryMgr<T>::Get().GetValue(ctx.device()));
  TRY_ASSIGN(auto kernel, (*f)->create(ctx.dtype()));
  kernel->compute(&ctx);
  return {};
}

Ret<BlobTensorPtr> AddFunctor::operator()(const BlobTensorPtr& a, const BlobTensorPtr& b) {
  CHECK_OR_RETURN(a->dtype() == b->dtype())
      << fmt::format("Tensor a({}) and Tensor b({}) must be same dtype.", a->dtype(), b->dtype());

  auto tc = std::shared_ptr<BlobTensor>(new CpuTensor(a->dtype(), a->shape()));
  KernelComputeContext ctx(a->device(), a->dtype());

  ctx.insertTensor("in", 0, std::move(a));
  ctx.insertTensor("in", 1, std::move(b));
  ctx.insertTensor("out", 0, tc);
  TRY(Call<AddKernelFactory>(ctx));
  return tc;
}

Ret<BlobTensorPtr> CompactFunctor::operator()(const BlobTensorPtr& a) {
  auto r = DeriveEmptyTensorLike(a);
  CHECK_OR_RETURN(r);
  KernelComputeContext ctx(a->device(), a->dtype());
  ctx.insertTensor("in", 0, std::move(a));
  ctx.insertTensor("out", 0, r);
  TRY(Call<CompactKernelFactory>(ctx));
  return r;
}

template <>
Ret<BlobTensorPtr> FillFunctor<const BlobTensorPtr&>::operator()(BlobTensorPtr& dst, const BlobTensorPtr& scalar) {
  CHECK_OR_RETURN(scalar->isScalar()) << "Tensor scalar is not scalar. shape: " << scalar->shape();
  CHECK_OR_RETURN(dst->dtype() == scalar->dtype())
      << fmt::format("Tensor dst({}) and Tensor scalar({}) must be same dtype.", dst->dtype(), scalar->dtype());
  KernelComputeContext ctx(dst->device(), dst->dtype());
  ctx.insertTensor("dst", 0, dst);
  ctx.insertTensor("scalar", 0, scalar);
  TRY(Call<FillKernelFactory>(ctx));
  return dst;
}

template <class T>
Ret<BlobTensorPtr> FillFunctor<T>::operator()(BlobTensorPtr& dst, T scalar) {
  CHECK_OR_RETURN(dst->dtype() == GetDataType<T>::value)
      << fmt::format("Tensor dst({}) and scalar({}) must be same dtype.", dst->dtype(), GetDataType<T>::value);
  KernelComputeContext ctx(dst->device(), dst->dtype());
  ctx.insertTensor("dst", 0, dst);
  ctx.insertTensor("scalar", 0, DeriveScalarOnSameDevice(dst, scalar));
  TRY(Call<FillKernelFactory>(ctx));
  return dst;
}

}  // namespace fineflow
