#ifndef FINEFLOW_CORE_FUNCTIONAL_IMPL_BASIC_FUNCTOR_H_
#define FINEFLOW_CORE_FUNCTIONAL_IMPL_BASIC_FUNCTOR_H_
#include "fineflow/core/common/data_type.h"
#include "fineflow/core/common/preprocess.h"
#include "fineflow/core/common/registry.h"
#define REGISTER_FUNCTOR(functor, key)                                                      \
  /* NOLINTBEGIN */                                                                         \
  REGISTER_KEY_WITH_CLASS_T(std::string, std::function<FuncType<functor>>, FunctorTag, key) \
      .setValue(std::function<FuncType<functor>>(functor()));                               \
  /* NOLINTEND */

#define REGISTER_FILL_FUNCTOR(type_cpp, type_proto) REGISTER_FUNCTOR(FillFunctor<type_cpp>, "fill")
#define MAP_REGISTER_FULL_FUNCTOR(tuple) FF_PP_FORWARD(REGISTER_FILL_FUNCTOR, FF_TUPLE_TO_ENUM(tuple))

#define REGISTER_ASSIGN_FUNCTOR(type_cpp, type_proto) REGISTER_FUNCTOR(AssignFunctor<type_cpp>, "assign")
#define MAP_REGISTER_ASSIGN_FUNCTOR(tuple) FF_PP_FORWARD(REGISTER_ASSIGN_FUNCTOR, FF_TUPLE_TO_ENUM(tuple))
#endif  // FINEFLOW_CORE_FUNCTIONAL_IMPL_BASIC_FUNCTOR_H_
