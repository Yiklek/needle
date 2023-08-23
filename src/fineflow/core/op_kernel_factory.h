#ifndef FINEFLOW_CORE_OP_KERNEL_FACTORY_H_
#define FINEFLOW_CORE_OP_KERNEL_FACTORY_H_
#include <functional>
#include <map>
#include <memory>

#include "fineflow/core/common/util.h"
namespace fineflow {
template <typename T>
class Factory {
public:
  FF_DISALLOW_COPY_AND_MOVE(Factory);
  Factory() = default;
  virtual ~Factory() = default;

  using Target = T;
};

template <typename Extend, typename T>
class OpKernelFactory : public Factory<T> {
public:
  FF_DISALLOW_COPY_AND_MOVE(OpKernelFactory);
  OpKernelFactory() = default;
  virtual ~OpKernelFactory() = default;

  using Target = T;
  template <typename... Args>
  std::unique_ptr<Target> create(Args&&... args) {
    return static_cast<Extend*>(this)->create(std::forward<Args>(args)...);
  };
};

template <typename T, typename D>
std::unique_ptr<T> NewKernalFromHandlers(const std::map<D, std::function<std::unique_ptr<T>()>>& handlers,
                                         const D& key) {
  const auto iter = handlers.find(key);
  if (iter != handlers.end()) {
    return iter->second();
  }
  return nullptr;
}

}  // namespace fineflow

#endif  // FINEFLOW_CORE_OP_KERNEL_FACTORY_H_
