module;

#include "fineflow/core/common/log.h"
#include "fineflow/core/common/result.h"
#include "fineflow/core/common/util.h"
export module fineflow.core.op_kernel_factory;
import fineflow.core.op_kernel;
import fineflow.core.common.data_type_proto;
import fineflow.core.common.device_type_proto;
import fineflow.core.common.log;
import fineflow.core.common.error;
import fineflow.core.common.registry_manager;
import std;

export namespace fineflow {
template <typename Extend, typename T>
class Factory {
public:
  FF_DISALLOW_COPY_AND_MOVE(Factory);
  Factory() = default;
  ~Factory() = default;

  using Target = T;
  using FactoryClass = Extend;
  template <typename Self, typename... Args>
  Ret<std::unique_ptr<Target>> create(this Self&& self, Args&&... args) {
    return std::forward_like<Self>().create(std::forward_like<Args>(args)...);
  };
};

class OpKernelFactory : public Factory<OpKernelFactory, OpKernel> {
public:
  FF_DISALLOW_COPY_AND_MOVE(OpKernelFactory);
  OpKernelFactory() = default;
  ~OpKernelFactory() = default;

  virtual Ret<std::unique_ptr<OpKernel>> create(DataType) { return UNIMPLEMENTED_ERROR; };
};

template <class T>
using KernelFactoryRegistryMgr = RegistryMgr<DeviceType, std::unique_ptr<OpKernelFactory>, T>;
}  // namespace fineflow
