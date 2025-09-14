module;
#include "fineflow/core/common/result.h"
export module fineflow.core.functional;
import std;
import fineflow.core.common.error;
import fineflow.core.common.registry_manager;
// #include <string>
//
// #include "fineflow/core/common/error_util.h"
// #include "fineflow/core/common/registry_manager.hpp"

export namespace fineflow {
template <class R, class... Args>
struct Functor {
  // functor on construct
  explicit Functor(const std::string &name)
      : name_(name), f_(RegistryFuncMgr::Get().GetValue(name_).value_or(nullptr)) {}
  explicit Functor(std::string &&name)
      : name_(std::move(name)), f_(RegistryFuncMgr::Get().GetValue(name_).value_or(nullptr)) {}
  using ReturnType = Ret<R>;

  // core func type
  using FuncType = std::function<ReturnType(Args...)>;
  using RegistryFuncMgr = RegistryMgr<std::string, FuncType>;
  ReturnType operator()(Args... args) {
    CHECK_OR_RETURN(f_) << "functor (" << name_ << ") is not registered.";
    return (*f_)(args...);
  }

protected:
  std::string name_;
  const FuncType *f_;
};

}  // namespace fineflow
