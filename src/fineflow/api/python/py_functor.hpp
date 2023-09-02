#ifndef FINEFLOW_API_PYTHON_PY_FUNCTOR_HPP_
#define FINEFLOW_API_PYTHON_PY_FUNCTOR_HPP_
#include <tuple>

#include "fineflow/api/python/py_tensor.h"
#include "fineflow/core/common/error_util.h"
namespace fineflow {

template <class... Args>
inline auto MapFunctorType(Args &&...args) {
  return std::tuple<IfElseT<std::is_same_v<Args, const python_api::Tensor &>, const BlobTensorPtr &, Args>...>(
      std::forward<IfElseT<std::is_same_v<Args, const python_api::Tensor &>, const BlobTensorPtr &, Args>>(args)...);
  // return std::tuple<IfElseT<std::is_same_v<Args, const python_api::Tensor &>, const BlobTensorPtr &,
  // Args>...>(args...);
}
template <class R, class... Args>
struct PyFunctor {
  explicit PyFunctor(std::string name) : name(std::move(name)) {}
  using FuncRet = IfElseT<std::is_same_v<R, python_api::Tensor>, BlobTensorPtr, R>;

  template <std::size_t... I>
  R apply(Args &&...args, std::index_sequence<I...>) {
    const python_api::Tensor t1 = python_api::Tensor::New(DeviceType::kCPU, 1);
    const python_api::Tensor t2 = python_api::Tensor::New(DeviceType::kCPU, 1);
    auto _xxx = MapFunctorType(t1, t2);
    auto mapped_args = MapFunctorType<Args...>(args...);
    using FuncArgs = decltype(mapped_args);
    using FunctorType = std::function<FuncRet(typename std::tuple_element<I, FuncArgs>::type...)>;
    TRY_CATCH(FF_PP_ALL(RegistryMgr<std::string, FunctorType>::Get().GetValue(name)),
              { LOG(fineflow::err) << fineflow::FormatErrorStr(e.stackedError()).value(); })
    return (**RegistryMgr<std::string, FunctorType>::Get().GetValue(name))(args...);
  }
  R operator()(Args... args) {
    auto indexes = std::make_index_sequence<sizeof...(args)>();
    return apply(args..., indexes);
  }
  std::string name;
};
}  // namespace fineflow
#endif  // FINEFLOW_API_PYTHON_PY_FUNCTOR_HPP_
