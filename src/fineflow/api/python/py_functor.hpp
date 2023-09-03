#ifndef FINEFLOW_API_PYTHON_PY_FUNCTOR_HPP_
#define FINEFLOW_API_PYTHON_PY_FUNCTOR_HPP_
#include <tuple>

#include "fineflow/api/python/py_tensor.h"
#include "fineflow/core/common/error_util.h"
#include "fineflow/core/common/result.hpp"
namespace fineflow {

template <class From, class To = From>
inline To Map(From f) {
  return f;
}

template <class From = const python_api::Tensor &,
          class = std::enable_if_t<std::is_same_v<From, const python_api::Tensor &>>>
inline const BlobTensorPtr &Map(const python_api::Tensor &f) {
  return f.ptr();
};

template <class... Args>
inline auto MapFunctorType(Args... args) {
  return std::tuple<decltype(Map<Args>(args))...>(Map<Args>(args)...);
}

template <class R, class... Args>
struct PyFunctor {
  explicit PyFunctor(std::string name) : name(std::move(name)) {}
  using FuncRet = IfElseT<std::is_same_v<R, python_api::Tensor>, Ret<BlobTensorPtr>, R>;

  template <std::size_t... I>
  inline R apply(Args &&...args, std::index_sequence<I...>) {
    auto mapped_args = MapFunctorType<Args...>(args...);
    using FuncArgs = decltype(mapped_args);
    using FunctorType = std::function<FuncRet(typename std::tuple_element<I, FuncArgs>::type...)>;
    using RegistryFunctorMgr = RegistryMgr<std::string, FunctorType>;
    TRY_ASSIGN_CATCH(auto f, RegistryFunctorMgr::Get().GetValue(name),
                     { throw std::invalid_argument(fineflow::FormatErrorStr(e.stackedError()).value()); })
    TRY_ASSIGN_CATCH(auto r, std::apply((*f), mapped_args),
                     { throw std::runtime_error(fineflow::FormatErrorStr(e.stackedError()).value()); })
    return r;
  }

  R operator()(Args... args) {
    auto indexes = std::make_index_sequence<sizeof...(args)>();
    return apply(args..., indexes);
  }
  std::string name;
};
}  // namespace fineflow
#endif  // FINEFLOW_API_PYTHON_PY_FUNCTOR_HPP_
