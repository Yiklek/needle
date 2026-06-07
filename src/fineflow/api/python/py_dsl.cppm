module;
#include "fineflow/core/common/result.h"

export module fineflow.api.python.py_dsl;

import std;
import fineflow.core.op_kernel;
import fineflow.core.op_kernel_factory;
import fineflow.core.blob_tensor;
import fineflow.core.common.device_type_proto;
import fineflow.core.common.registry_manager;
import fineflow.core.common.error;
import fineflow.core.kernels.dsl.dsl_kernel;
import fineflow.core.kernels.dsl.device_launcher;
import fineflow.core.kernels.dsl.dsl_registry;

export namespace fineflow::python_api {

// Register a DSL kernel with a C++ compute function.
// The compute_fn receives a KernelComputeContext and operates on tensors directly.
inline void RegisterDSLKernelCpp(
    const std::string& name,
    DeviceType device,
    std::function<void(KernelComputeContext&)> compute_fn
) {
  namespace dsl = fineflow::dsl;

  dsl::DSLKernelMeta meta;
  meta.name = name;
  meta.source_type = dsl::Source::kTileLang;
  meta.target_device = device;
  meta.cpu_compute = std::move(compute_fn);

  (void)dsl::DSLKernelRegistry::Register(std::move(meta));
}

}  // namespace fineflow::python_api
