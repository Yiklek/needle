#ifndef FINEFLOW_CORE_KERNELS_CPU_KERNEL_FACTOR_HPP_
#define FINEFLOW_CORE_KERNELS_CPU_KERNEL_FACTOR_HPP_
#include "fineflow/core/common/data_type.h"
#include "fineflow/core/common/map.h"
#include "fineflow/core/common/preprocess.h"
#include "fineflow/core/common/registry.h"

#define REGISTER_KERNEL_FACTORY(device, kernel_factory_type) \
  REGISTER_KEY_VALUE(device, std::make_unique<kernel_factory_type>());

#define MAKE_NEW_FACTORY_ITEM_IMPL(type_cpp, type_proto, name) \
  { type_proto, name<type_cpp> }
#define MAKE_NEW_FACTORY_ITEM(type, name) FF_PP_FORWARD(MAKE_NEW_FACTORY_ITEM_IMPL, FF_TUPLE_TO_ENUM(type), name)
#define MAKE_NEW_FACTORY(name) MAP_LIST_UD(MAKE_NEW_FACTORY_ITEM, name, CPU_PRIMITIVE_NATIVE_TYPE_ENUM)

#endif  // FINEFLOW_CORE_KERNELS_CPU_KERNEL_FACTOR_HPP_
