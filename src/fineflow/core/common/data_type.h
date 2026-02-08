#ifndef FINEFLOW_CORE_COMMON_DATA_TYPE_H_
#define FINEFLOW_CORE_COMMON_DATA_TYPE_H_

#include "map.h"
#include "preprocess.h"
#include "registry.h"

#define FF_TUPLE_TO_ENUM_IMPL(...) __VA_ARGS__
#define FF_TUPLE_TO_ENUM(...) FF_TUPLE_TO_ENUM_IMPL __VA_ARGS__
#define FF_LIST_TO_TUPLE(...) (__VA_ARGS__)
#define FF_TUPLE_TO_SEQ(...) MAP(FF_PP_ALL, FF_TUPLE_TO_ENUM(__VA_ARGS__))

#define CPU_PRIMITIVE_BOOL_TYPE_TUPLE FF_LIST_TO_TUPLE(bool, DataType::kBool)
#define CPU_PRIMITIVE_CHAR_TYPE_TUPLE FF_LIST_TO_TUPLE(char, DataType::kChar)
#define CPU_PRIMITIVE_INT8_TYPE_TUPLE FF_LIST_TO_TUPLE(int8_t, DataType::kInt8)
#define CPU_PRIMITIVE_UINT8_TYPE_TUPLE FF_LIST_TO_TUPLE(uint8_t, DataType::kUInt8)
#define CPU_PRIMITIVE_INT16_TYPE_TUPLE FF_LIST_TO_TUPLE(int16_t, DataType::kInt16)
#define CPU_PRIMITIVE_UINT16_TYPE_TUPLE FF_LIST_TO_TUPLE(uint16_t, DataType::kUInt16)
#define CPU_PRIMITIVE_INT32_TYPE_TUPLE FF_LIST_TO_TUPLE(int32_t, DataType::kInt32)
#define CPU_PRIMITIVE_UINT32_TYPE_TUPLE FF_LIST_TO_TUPLE(uint32_t, DataType::kUInt32)
#define CPU_PRIMITIVE_INT64_TYPE_TUPLE FF_LIST_TO_TUPLE(int64_t, DataType::kInt64)
#define CPU_PRIMITIVE_UINT64_TYPE_TUPLE FF_LIST_TO_TUPLE(uint64_t, DataType::kUInt64)
#define CPU_PRIMITIVE_FLOAT_TYPE_TUPLE FF_LIST_TO_TUPLE(float, DataType::kFloat)
#define CPU_PRIMITIVE_DOUBLE_TYPE_TUPLE FF_LIST_TO_TUPLE(double, DataType::kDouble)
#define CPU_PRIMITIVE_COMPLEX_FLOAT_TYPE_TUPLE FF_LIST_TO_TUPLE(std::complex<float>, DataType::kComplex64)
#define CPU_PRIMITIVE_COMPLEX_DOUBLE_TYPE_TUPLE FF_LIST_TO_TUPLE(std::complex<double>, DataType::kComplex128)

#define CPU_PRIMITIVE_FLOAT16_TYPE_TUPLE FF_LIST_TO_TUPLE(float16, DataType::kFloat16)
#define CPU_PRIMITIVE_BFLOAT16_TYPE_TUPLE FF_LIST_TO_TUPLE(bfloat16, DataType::kBFloat16)

#define CPU_PRIMITIVE_NATIVE_TYPE_TUPLE                                                                              \
  FF_LIST_TO_TUPLE(CPU_PRIMITIVE_BOOL_TYPE_TUPLE, CPU_PRIMITIVE_CHAR_TYPE_TUPLE, CPU_PRIMITIVE_INT8_TYPE_TUPLE,      \
                   CPU_PRIMITIVE_UINT8_TYPE_TUPLE, CPU_PRIMITIVE_INT16_TYPE_TUPLE, CPU_PRIMITIVE_UINT16_TYPE_TUPLE,  \
                   CPU_PRIMITIVE_INT32_TYPE_TUPLE, CPU_PRIMITIVE_UINT32_TYPE_TUPLE, CPU_PRIMITIVE_INT64_TYPE_TUPLE,  \
                   CPU_PRIMITIVE_UINT64_TYPE_TUPLE, CPU_PRIMITIVE_FLOAT_TYPE_TUPLE, CPU_PRIMITIVE_DOUBLE_TYPE_TUPLE, \
                   CPU_PRIMITIVE_COMPLEX_FLOAT_TYPE_TUPLE, CPU_PRIMITIVE_COMPLEX_DOUBLE_TYPE_TUPLE)

#define CPU_PRIMITIVE_NATIVE_TYPE_SEQ FF_TUPLE_TO_SEQ(CPU_PRIMITIVE_NATIVE_TYPE_TUPLE)
#define CPU_PRIMITIVE_NATIVE_TYPE_ENUM FF_TUPLE_TO_ENUM(CPU_PRIMITIVE_NATIVE_TYPE_TUPLE)

#define SPECIALIZE_GET_DATA_TYPE(type_cpp, type_proto)                          \
  template <>                                                                   \
  struct GetDataType<type_cpp> : std::integral_constant<DataType, type_proto> { \
    static constexpr size_t size = sizeof(type_cpp);                            \
  };                                                                            \
  template <>                                                                   \
  struct DataTypeToClass<type_proto> : type_identity<type_cpp> {                \
    static constexpr size_t size = sizeof(type_cpp);                            \
  };

#define MAP_SPECIALIZE_GET_DATA_TYPE(tuple) FF_PP_FORWARD(SPECIALIZE_GET_DATA_TYPE, FF_TUPLE_TO_ENUM(tuple))
#define REGISTER_TYPE(type_cpp, type_proto) REGISTER_KEY_VALUE_MGR(DataTypeSizeRegistryMgr, type_proto, sizeof(type_cpp));
#define MAP_REGISTER_TYPE(tuple) FF_PP_FORWARD(REGISTER_TYPE, FF_TUPLE_TO_ENUM(tuple))

#endif  // FINEFLOW_CORE_COMMON_DATA_TYPE_H_
