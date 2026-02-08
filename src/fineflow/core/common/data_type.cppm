module;
#include "fineflow/core/common/data_type.h"
export module fineflow.core.common.data_type;

import std;
import std.compat;
import fineflow.core.common.data_type_proto;
import fineflow.core.common.registry_manager;
import fineflow.core.common.util;

export namespace fineflow {

template <typename T, typename T2 = void>
struct GetDataType;

template <DataType Type>
struct DataTypeToClass;

template <>
struct GetDataType<void> : std::integral_constant<DataType, DataType::kChar> {};

template <DataType Type>
using DataTypeToType = typename DataTypeToClass<Type>::type;

struct DataTypeToSizeTag{};
using DataTypeSizeRegistryMgr = RegistryMgr<DataType, size_t, DataTypeToSizeTag>;

template <typename T>
using type_identity = std::type_identity<T>;

MAP(MAP_SPECIALIZE_GET_DATA_TYPE, FF_TUPLE_TO_ENUM(CPU_PRIMITIVE_NATIVE_TYPE_TUPLE))
}  // namespace fineflow
namespace fineflow {
MAP(MAP_REGISTER_TYPE, FF_TUPLE_TO_ENUM(CPU_PRIMITIVE_NATIVE_TYPE_TUPLE))
}
