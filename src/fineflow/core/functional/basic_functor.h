#ifndef FINEFLOW_CORE_FUNCTIONAL_BASIC_FUNCTOR_H_
#define FINEFLOW_CORE_FUNCTIONAL_BASIC_FUNCTOR_H_
#include "fineflow/core/blob_tensor.h"
#include "fineflow/core/common/function_traits.hpp"
#include "fineflow/core/common/registry_manager.hpp"
namespace fineflow {

#define REGISTER_FUNCTOR(functor, key)                                        \
  /* NOLINTBEGIN */                                                           \
  REGISTER_KEY_WITH_CLASS(std::string, std::function<FuncType<functor>>, key) \
      .setValue(std::function<FuncType<functor>>(functor()));                 \
  /* NOLINTEND */

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
}  // namespace fineflow
#endif  // FINEFLOW_CORE_FUNCTIONAL_BASIC_FUNCTOR_H_
