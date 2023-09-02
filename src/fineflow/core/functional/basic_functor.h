#ifndef FINEFLOW_CORE_FUNCTIONAL_BASIC_FUNCTOR_H_
#define FINEFLOW_CORE_FUNCTIONAL_BASIC_FUNCTOR_H_
#include "fineflow/core/blob_tensor.h"
#include "fineflow/core/common/function_traits.hpp"
#include "fineflow/core/common/registry_manager.hpp"

namespace fineflow {
class AddFunctor {
public:
  AddFunctor() = default;

  BlobTensorPtr operator()(const BlobTensorPtr& a, const BlobTensorPtr& b);
};
namespace {
using AddFunctorType = function_traits<AddFunctor>::func_type;
REGISTER_KEY(std::function<AddFunctorType>, std::string("add")).setValue(std::function<AddFunctorType>(AddFunctor()));
}  // namespace
}  // namespace fineflow
#endif  // FINEFLOW_CORE_FUNCTIONAL_BASIC_FUNCTOR_H_
