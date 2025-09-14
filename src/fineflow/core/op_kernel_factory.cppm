module;

#include "fineflow/core/common/util.h"
export module fineflow.core.op_kernel_factory;
import std;

export namespace fineflow {
template <typename T>
class Factory {
public:
  FF_DISALLOW_COPY_AND_MOVE(Factory);
  Factory() = default;
  ~Factory() = default;

  using Target = T;
};

template <typename Extend, typename T>
class OpKernelFactory : public Factory<T> {
public:
  FF_DISALLOW_COPY_AND_MOVE(OpKernelFactory);
  OpKernelFactory() = default;
  ~OpKernelFactory() = default;

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
