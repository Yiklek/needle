module;
#include "basic_functor.h"
#include "fineflow/core/common/result.h"
export module func_impl;
import fineflow.core.op_kernel.cpu.add_kernel;
import fineflow.core.op_kernel.cpu.assign_kernel;
import fineflow.core.op_kernel.cpu.compact_kernel;
import fineflow.core.op_kernel.cpu.fill_kernel;
import fineflow.core.op_kernel;
import fineflow.core.op_kernel_factory;
import fineflow.core.functional;
import fineflow.core.blob_tensor;
import fineflow.core.common.registry_manager;
import fineflow.core.common.function_traits;
import fineflow.core.common.error;
import fineflow.core.common.fmt;
import std;
import std.compat;
export namespace fineflow {

class AddFunctor {
public:
  AddFunctor() = default;

  Ret<BlobTensorView> operator()(const BlobTensorView& a, const BlobTensorView& b);
};
using AddFunctorType = FuncType<AddFunctor>;
class CompactFunctor {
public:
  CompactFunctor() = default;

  Ret<BlobTensorView> operator()(const BlobTensorView& a);
};
using CompactFunctorType = FuncType<CompactFunctor>;

template <class T>
class FillFunctor {
public:
  FillFunctor() = default;

  Ret<void> operator()(BlobTensorView& dst, T scalar);
};

template <class T>
class AssignFunctor {
public:
  AssignFunctor() = default;

  Ret<void> operator()(BlobTensorView& dst, T src);
};

template <class T>
inline Ret<void> Call(KernelComputeContext& ctx) {
  TRY_ASSIGN(auto f, KernelFactoryRegistryMgr<T>::Get().GetValue(ctx.device()));
  TRY_ASSIGN(auto kernel, (*f)->create(ctx.dtype()));
  kernel->compute(ctx);
  return {};
}

inline Ret<void> Call(const std::string& kernel_name, KernelComputeContext& ctx) {
  TRY_ASSIGN(auto f, RuntimeKernelFactoryRegistryMgr::Get().GetValue({kernel_name, ctx.device()}));
  TRY_ASSIGN(auto kernel, (*f)->create(ctx.dtype()));
  kernel->compute(ctx);
  return {};
}

Ret<BlobTensorView> AddFunctor::operator()(const BlobTensorView& a, const BlobTensorView& b) {
  CHECK_OR_RETURN(a.dtype() == b.dtype())
      << format("Tensor a({}) and Tensor b({}) must be same dtype.", a.dtype(), b.dtype());
  auto tc = DeriveEmptyTensorLike(a);

  KernelComputeContext ctx(a.device(), a.dtype());
  ctx.insertTensor("in", 0, a);
  ctx.insertTensor("in", 1, b);
  ctx.insertTensor("out", 0, tc->view());
  // TRY(Call<AddKernel>(ctx));
  TRY(Call("Add", ctx));
  return tc->view();
}

Ret<BlobTensorView> CompactFunctor::operator()(const BlobTensorView& a) {
  auto r = DeriveEmptyTensorLike(a);
  CHECK_OR_RETURN(r);
  KernelComputeContext ctx(a.device(), a.dtype());
  ctx.insertTensor("in", 0, a);
  ctx.insertTensor("out", 0, r->view());
  TRY(Call<CompactKernel>(ctx));
  return r->view();
}

template <>
Ret<void> FillFunctor<const BlobTensorView&>::operator()(BlobTensorView& dst, const BlobTensorView& scalar) {
  CHECK_OR_RETURN(scalar.isScalar()) << "Tensor scalar is not scalar. shape: " << scalar.shape();
  CHECK_OR_RETURN(dst.dtype() == scalar.dtype())
      << format("Tensor dst({}) and Tensor scalar({}) must be same dtype.", dst.dtype(), scalar.dtype());
  KernelComputeContext ctx(dst.device(), dst.dtype());
  ctx.insertTensor("dst", 0, dst);
  ctx.insertTensor("scalar", 0, scalar);
  TRY(Call<FillKernel>(ctx));
  return {};
}

template <class T>
Ret<void> FillFunctor<T>::operator()(BlobTensorView& dst, T scalar) {
  return FillFunctor<const BlobTensorView&>()(dst, DeriveScalarOnSameDevice(dst, scalar));
}

template <>
Ret<void> AssignFunctor<const BlobTensorView&>::operator()(BlobTensorView& dst, const BlobTensorView& src) {
  CHECK_OR_RETURN(src.isScalar() || dst.shape() == src.shape())
      << format("Tensor src should be scalar or src's shape is same as dst's. src shape: {}, dst shape: {}",
                src.shape(), dst.shape());
  CHECK_OR_RETURN(dst.dtype() == src.dtype())
      << format("Tensor dst({}) and Tensor src({}) must be same dtype.", dst.dtype(), src.dtype());
  KernelComputeContext ctx(dst.device(), dst.dtype());
  ctx.insertTensor("dst", 0, dst);
  ctx.insertTensor("src", 0, src);
  TRY(Call<AssignKernel>(ctx));
  return {};
}

template <class T>
Ret<void> AssignFunctor<T>::operator()(BlobTensorView& dst, T src) {
  return AssignFunctor<const BlobTensorView&>()(dst, DeriveScalarOnSameDevice(dst, src));
}

}  // namespace fineflow
namespace fineflow {
REGISTER_FUNCTOR(AddFunctor, "add")
REGISTER_FUNCTOR(CompactFunctor, "compact")
REGISTER_FUNCTOR(FillFunctor<const BlobTensorView&>, "fill")
MAP(MAP_REGISTER_FULL_FUNCTOR, CPU_PRIMITIVE_NATIVE_TYPE_ENUM)

REGISTER_FUNCTOR(AssignFunctor<const BlobTensorView&>, "assign")
MAP(MAP_REGISTER_ASSIGN_FUNCTOR, CPU_PRIMITIVE_NATIVE_TYPE_ENUM)

// #include "reg_functor.gen"
}  // namespace fineflow
