#ifndef FINEFLOW_CORE_COMMON_TYPES_TUPLE_H_
#define FINEFLOW_CORE_COMMON_TYPES_TUPLE_H_
#include <complex>

#include "fineflow/core/common/data_type.pb.h"
#include "fineflow/core/common/preprocess.h"
#include "fineflow/core/common/util.h"
#define CPU_PRIMITIVE_BOOL_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(bool, DataType::kBool)
#define CPU_PRIMITIVE_CHAR_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(char, DataType::kChar)
#define CPU_PRIMITIVE_INT8_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(int8_t, DataType::kInt8)
#define CPU_PRIMITIVE_UINT8_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(uint8_t, DataType::kUInt8)
#define CPU_PRIMITIVE_INT16_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(int16_t, DataType::kInt16)
#define CPU_PRIMITIVE_UINT16_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(uint16_t, DataType::kUInt16)
#define CPU_PRIMITIVE_INT32_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(int32_t, DataType::kInt32)
#define CPU_PRIMITIVE_UINT32_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(uint32_t, DataType::kUInt32)
#define CPU_PRIMITIVE_INT64_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(int64_t, DataType::kInt64)
#define CPU_PRIMITIVE_UINT64_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(uint64_t, DataType::kUInt64)
#define CPU_PRIMITIVE_FLOAT_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(float, DataType::kFloat)
#define CPU_PRIMITIVE_DOUBLE_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(double, DataType::kDouble)
#define CPU_PRIMITIVE_COMPLEX_FLOAT_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(std::complex<float>, DataType::kComplex64)
#define CPU_PRIMITIVE_COMPLEX_DOUBLE_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(std::complex<double>, DataType::kComplex128)

#define CPU_PRIMITIVE_FLOAT16_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(float16, DataType::kFloat16)
#define CPU_PRIMITIVE_BFLOAT16_TYPE_TUPLE BOOST_PP_VARIADIC_TO_TUPLE(bfloat16, DataType::kBFloat16)

#define CPU_PRIMITIVE_NATIVE_TYPE_TUPLE                                                                 \
  BOOST_PP_VARIADIC_TO_TUPLE(                                                                           \
      CPU_PRIMITIVE_BOOL_TYPE_TUPLE, CPU_PRIMITIVE_CHAR_TYPE_TUPLE, CPU_PRIMITIVE_INT8_TYPE_TUPLE,      \
      CPU_PRIMITIVE_UINT8_TYPE_TUPLE, CPU_PRIMITIVE_INT16_TYPE_TUPLE, CPU_PRIMITIVE_UINT16_TYPE_TUPLE,  \
      CPU_PRIMITIVE_INT32_TYPE_TUPLE, CPU_PRIMITIVE_UINT32_TYPE_TUPLE, CPU_PRIMITIVE_INT64_TYPE_TUPLE,  \
      CPU_PRIMITIVE_UINT64_TYPE_TUPLE, CPU_PRIMITIVE_FLOAT_TYPE_TUPLE, CPU_PRIMITIVE_DOUBLE_TYPE_TUPLE, \
      CPU_PRIMITIVE_COMPLEX_FLOAT_TYPE_TUPLE, CPU_PRIMITIVE_COMPLEX_DOUBLE_TYPE_TUPLE)

#define CPU_PRIMITIVE_NATIVE_TYPE_SEQ BOOST_PP_TUPLE_TO_SEQ(CPU_PRIMITIVE_NATIVE_TYPE_TUPLE)

namespace fineflow {
#define IS_CPU_NATIVE(t, type_native) (std::is_same_v<t, type_native>)
#define FOR_IS_CPU_NATIVE(i, data, elem) (IS_CPU_NATIVE(data, BOOST_PP_TUPLE_ELEM(0, elem)))
template <class T>
#define S BOOST_PP_SEQ_FOR_EACH(FOR_IS_CPU_NATIVE, T, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)
static constexpr bool is_cpu_native_v = Or(BOOST_PP_SEQ_ENUM(S));
}  // namespace fineflow
// generate from this:
// template <class T>
// static constexpr bool is_cpu_native_v = FF_PP_JOIN_SEQ(FF_PP_ALL(||), S);
//
// template <class T>
// static constexpr bool is_cpu_native_v =
//     ((std ::is_same_v<T, bool>)) || ((std ::is_same_v<T, char>)) || ((std ::is_same_v<T, int8_t>)) ||
//     ((std ::is_same_v<T, uint8_t>)) || ((std ::is_same_v<T, int16_t>)) || ((std ::is_same_v<T, uint16_t>)) ||
//     ((std ::is_same_v<T, int32_t>)) || ((std ::is_same_v<T, uint32_t>)) || ((std ::is_same_v<T, int64_t>)) ||
//     ((std ::is_same_v<T, uint64_t>)) || ((std ::is_same_v<T, float>)) || ((std ::is_same_v<T, double>)) ||
//     ((std ::is_same_v<T, std ::complex<float> >)) || ((std ::is_same_v<T, std ::complex<double> >));

#endif  // FINEFLOW_CORE_COMMON_TYPES_TUPLE_H_
