#include "fineflow/core/common/preprocess.h"
#include "fineflow/core/common/result.hpp"
#include "fineflow/core/cpu/cpu_tensor.h"
#include "fineflow/core/functional/basic_functor.h"
#include "fineflow/core/kernels/add_kernel.h"
#include "fineflow/core/kernels/assign_kernel.h"
#include "fineflow/core/kernels/compact_kernel.h"
#include "fineflow/core/kernels/fill_kernel.h"
#include "fineflow/core/tensor_util.h"

namespace fineflow {

namespace {
REGISTER_FUNCTOR(AddFunctor, "add")
REGISTER_FUNCTOR(CompactFunctor, "compact")
REGISTER_FUNCTOR(FillFunctor<const BlobTensorView&>, "fill")
#define REGISTER_FILL_FUNCTOR(type_cpp, type_proto) REGISTER_FUNCTOR(FillFunctor<type_cpp>, "fill")
#define FOR_REGISTER_FILL_FUNCTOR(i, data, elem) REGISTER_FILL_FUNCTOR elem
BOOST_PP_SEQ_FOR_EACH(FOR_REGISTER_FILL_FUNCTOR, _, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)
#undef FOR_REGISTER_FULL_FUNCTOR
#undef REGISTER_FILL_FUNCTOR

REGISTER_FUNCTOR(AssignFunctor<const BlobTensorView&>, "assign")
#define REGISTER_ASSIGN_FUNCTOR(type_cpp, type_proto) REGISTER_FUNCTOR(AssignFunctor<type_cpp>, "assign")
#define FOR_REGISTER_ASSIGN_FUNCTOR(i, data, elem) REGISTER_ASSIGN_FUNCTOR elem
BOOST_PP_SEQ_FOR_EACH(FOR_REGISTER_ASSIGN_FUNCTOR, _, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)
#undef FOR_REGISTER_FULL_FUNCTOR
#undef REGISTER_ASSIGN_FUNCTOR

}  // namespace

template <class T>
inline Ret<void> Call(KernelComputeContext& ctx) {
  TRY_ASSIGN(auto f, KernelRegistryMgr<T>::Get().GetValue(ctx.device()));
  TRY_ASSIGN(auto kernel, (*f)->create(ctx.dtype()));
  kernel->compute(&ctx);
  return {};
}

Ret<BlobTensorView> AddFunctor::operator()(const BlobTensorView& a, const BlobTensorView& b) {
  CHECK_OR_RETURN(a.dtype() == b.dtype())
      << fmt::format("Tensor a({}) and Tensor b({}) must be same dtype.", a.dtype(), b.dtype());
  auto tc = DeriveEmptyTensorLike(a);

  KernelComputeContext ctx(a.device(), a.dtype());
  ctx.insertTensor("in", 0, a);
  ctx.insertTensor("in", 1, b);
  ctx.insertTensor("out", 0, tc->view());
  TRY(Call<AddKernelFactory>(ctx));
  return tc->view();
}

Ret<BlobTensorView> CompactFunctor::operator()(const BlobTensorView& a) {
  auto r = DeriveEmptyTensorLike(a);
  CHECK_OR_RETURN(r);
  KernelComputeContext ctx(a.device(), a.dtype());
  ctx.insertTensor("in", 0, a);
  ctx.insertTensor("out", 0, r->view());
  TRY(Call<CompactKernelFactory>(ctx));
  return r->view();
}

template <>
Ret<void> FillFunctor<const BlobTensorView&>::operator()(BlobTensorView& dst, const BlobTensorView& scalar) {
  CHECK_OR_RETURN(scalar.isScalar()) << "Tensor scalar is not scalar. shape: " << scalar.shape();
  CHECK_OR_RETURN(dst.dtype() == scalar.dtype())
      << fmt::format("Tensor dst({}) and Tensor scalar({}) must be same dtype.", dst.dtype(), scalar.dtype());
  KernelComputeContext ctx(dst.device(), dst.dtype());
  ctx.insertTensor("dst", 0, dst);
  ctx.insertTensor("scalar", 0, scalar);
  TRY(Call<FillKernelFactory>(ctx));
  return {};
}

template <class T>
Ret<void> FillFunctor<T>::operator()(BlobTensorView& dst, T scalar) {
  return FillFunctor<const BlobTensorView&>()(dst, DeriveScalarOnSameDevice(dst, scalar)->view());
}

template <>
Ret<void> AssignFunctor<const BlobTensorView&>::operator()(BlobTensorView& dst, const BlobTensorView& src) {
  CHECK_OR_RETURN(src.isScalar() || dst.shape() == src.shape())
      << fmt::format("Tensor src should be scalar or src's shape is same as dst's. src shape: {}, dst shape: {}",
                     src.shape(), dst.shape());
  CHECK_OR_RETURN(dst.dtype() == src.dtype())
      << fmt::format("Tensor dst({}) and Tensor src({}) must be same dtype.", dst.dtype(), src.dtype());
  KernelComputeContext ctx(dst.device(), dst.dtype());
  ctx.insertTensor("dst", 0, dst);
  ctx.insertTensor("src", 0, src);
  TRY(Call<AssignKernelFactory>(ctx));
  return {};
}

template <class T>
Ret<void> AssignFunctor<T>::operator()(BlobTensorView& dst, T src) {
  return AssignFunctor<const BlobTensorView&>()(dst, DeriveScalarOnSameDevice(dst, src)->view());
}

}  // namespace fineflow
