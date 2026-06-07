module;
#include "fineflow/core/common/result.h"

export module fineflow.core.kernels.dsl.dsl_registry;

import std;
import fineflow.core.kernels.dsl.dsl_kernel;
import fineflow.core.kernels.dsl.device_launcher;
import fineflow.core.op_kernel_factory;
import fineflow.core.common.device_type_proto;
import fineflow.core.common.registry_manager;
import fineflow.core.common.error;

export namespace fineflow::dsl {

struct DSLKernelRegistry {
  static Ret<void> Register(DSLKernelMeta meta) {
    auto key = std::make_pair(meta.name, meta.target_device);
    auto factory = std::make_unique<DSLOpKernelFactory>(std::move(meta));
    return RuntimeKernelFactoryRegistryMgr::Get().Register(std::move(key), std::move(factory));
  }
};

}  // namespace fineflow::dsl
