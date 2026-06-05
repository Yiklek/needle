#ifndef FINEFLOW_CORE_KERNELS_CPU_KERNEL_FACTOR_HPP_
#define FINEFLOW_CORE_KERNELS_CPU_KERNEL_FACTOR_HPP_
#include "fineflow/core/common/data_type.h"
#include "fineflow/core/common/map.h"
#include "fineflow/core/common/preprocess.h"
#include "fineflow/core/common/registry.h"

#define MAKE_NEW_FACTORY_ITEM_IMPL(type_cpp, type_proto, name) {type_proto, NewOpKernel<name##KernelImpl<type_cpp>>}
#define MAKE_NEW_FACTORY_ITEM(type, name) FF_PP_FORWARD(MAKE_NEW_FACTORY_ITEM_IMPL, FF_TUPLE_TO_ENUM(type), name)
#define MAKE_NEW_FACTORY(name) MAP_LIST_UD(MAKE_NEW_FACTORY_ITEM, name, CPU_PRIMITIVE_NATIVE_TYPE_ENUM)

#define REGISTER_KERNEL_FACTORY(kernel_name, device)                                                      \
  REGISTER_KEY_WITH_CLASS_T(DeviceType, std::unique_ptr<OpKernelFactory>, kernel_name##Kernel, device)    \
      .setValue(std::make_unique<kernel_name##KernelFactory>());                                          \
  REGISTER_KEY_WITH_CLASS_T(RuntimeKernelFactoryRegistryMgr::Key, RuntimeKernelFactoryRegistryMgr::Value, \
                            RuntimeKernelFactoryRegistryMgr::Tag,                                        \
                            FF_PP_ALL(std::pair<std::string, DeviceType>{#kernel_name, device}))          \
      .setValue(std::make_unique<kernel_name##KernelFactory>());

#define DECL_KERNEL(kernel_name)                                    \
  class kernel_name##KernelFactory;                                 \
  class kernel_name##Kernel : public OpKernel {                     \
  public:                                                           \
    FF_DISALLOW_COPY_AND_MOVE(kernel_name##Kernel);                 \
    kernel_name##Kernel() = default;                                \
  };                                                                \
  class kernel_name##KernelFactory final : public OpKernelFactory { \
  public:                                                           \
    Ret<std::unique_ptr<OpKernel>> create(DataType dtype) override; \
  };

// template <typename T>                                                                         \
// std::unique_ptr<OpKernel> New##kernel_name() {                                                \
//   return std::make_unique<kernel_name##KernelImpl<T>>();                                      \
// }                                                                                             \

#define IMPL_KERNEL_FACTORY(kernel_name)                                                        \
  Ret<std::unique_ptr<OpKernel>> kernel_name##KernelFactory::create(DataType dtype) {           \
    static const std::map<DataType, std::function<std::unique_ptr<OpKernel>()>> new_add_handle{ \
        MAKE_NEW_FACTORY(kernel_name)};                                                         \
    auto kernel = NewKernalFromHandlers(new_add_handle, dtype);                                 \
    CHECK_OR_RETURN(kernel) << #kernel_name << "Kernel for type: " << std::to_string(dtype)     \
                            << " has not implemented.";                                         \
    return kernel;                                                                              \
  };

#endif  // FINEFLOW_CORE_KERNELS_CPU_KERNEL_FACTOR_HPP_
