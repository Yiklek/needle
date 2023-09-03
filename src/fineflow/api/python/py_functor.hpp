#ifndef FINEFLOW_API_PYTHON_PY_FUNCTOR_HPP_
#define FINEFLOW_API_PYTHON_PY_FUNCTOR_HPP_
#include <tuple>

#include "fineflow/api/python/py_tensor.h"
#include "fineflow/core/common/error_util.h"
#include "fineflow/core/common/result.hpp"
namespace fineflow::python_api {

template <class T>
struct MapRetType : public type_identity<T> {};

template <>
struct MapRetType<python_api::Tensor> {
  using type = Ret<BlobTensorPtr>;
};

template <class T>
using MapRetTypeT = typename MapRetType<T>::type;

template <class T>
struct MapArgType : public type_identity<T> {};

template <>
struct MapArgType<const Tensor &> {
  using type = const BlobTensorPtr &;
};

template <class T>
using MapArgTypeT = typename MapArgType<T>::type;

/**
 * @brief Map python api arg to core functor arg
 * If arg connot be convert automatically, template specialization must be implemented.
 *
 * @tparam From Python api type.
 * @param f Python api arg.
 * @return Core functor arg.
 */
template <class From>
inline MapArgTypeT<From> MapArg(From f) {
  return f;
}

/**
 * @brief Same to MapArg. But map core functor result to python api result.
 * If arg connot be convert automatically, template specialization must be implemented.
 *
 * @tparam From Core functor type.
 * @param f Core functor result.
 * @return Python api result.
 */
template <class From>
inline MapRetTypeT<From> MapRet(From f) {
  return f;
}

template <class... Args>
inline auto MapArgs(Args... args) {
  return std::tuple<MapArgTypeT<Args>...>(MapArg<Args>(args)...);
}

template <class R, class... Args>
struct PyFunctor {
  explicit PyFunctor(const std::string &name) : name_(name) {}
  explicit PyFunctor(std::string &&name) : name_(std::move(name)) {}

  // core func type
  using FuncType = std::function<MapRetTypeT<R>(MapArgTypeT<Args>...)>;
  using RegistryFuncMgr = RegistryMgr<std::string, FuncType>;

  R operator()(Args... args) {
    auto mapped_args = MapArgs<Args...>(args...);
    TRY_ASSIGN_CATCH(auto f, RegistryFuncMgr::Get().GetValue(name_),
                     { throw std::invalid_argument(fineflow::FormatErrorStr(e.stackedError()).value()); })
    TRY_ASSIGN_CATCH(auto r, std::apply((*f), mapped_args),
                     { throw std::runtime_error(fineflow::FormatErrorStr(e.stackedError()).value()); })
    return MapRet(r);
  }

private:
  std::string name_;
};
}  // namespace fineflow::python_api
#endif  // FINEFLOW_API_PYTHON_PY_FUNCTOR_HPP_
