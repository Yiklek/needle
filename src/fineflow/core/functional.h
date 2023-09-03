#ifndef FINEFLOW_CORE_FUNCTIONAL_H_
#define FINEFLOW_CORE_FUNCTIONAL_H_
#include <string>

#include "fineflow/core/common/error_util.h"
#include "fineflow/core/common/registry_manager.hpp"
namespace fineflow {
template <class R, class... Args>
struct Functor {
  explicit Functor(const std::string &name) : name_(name) {}
  explicit Functor(std::string &&name) : name_(std::move(name)) {}

  // core func type
  using FuncType = std::function<R(Args...)>;
  using RegistryFuncMgr = RegistryMgr<std::string, FuncType>;

  R operator()(Args... args) {
    TRY_ASSIGN(auto f, RegistryFuncMgr::Get().GetValue(name_))
    return (*f)(args...);
  }

protected:
  std::string name_;
};

}  // namespace fineflow

#endif  // FINEFLOW_CORE_FUNCTIONAL_H_
