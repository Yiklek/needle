#ifndef FINEFLOW_CORE_FUNCTIONAL_H_
#define FINEFLOW_CORE_FUNCTIONAL_H_
#include <string>

#include "fineflow/core/common/error_util.h"
#include "fineflow/core/common/registry_manager.hpp"
#include "fineflow/core/kernels/add_kernel.h"
namespace fineflow {
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

#endif  // FINEFLOW_CORE_FUNCTIONAL_H_
