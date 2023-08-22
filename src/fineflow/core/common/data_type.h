#ifndef FINEFLOW_CORE_COMMON_DATA_TYPE_H_
#define FINEFLOW_CORE_COMMON_DATA_TYPE_H_
#include <type_traits>

#include "fineflow/core/common/data_type.pb.h"
namespace fineflow {

template <typename T, typename T2 = void>
struct GetDataType;

template <>
struct GetDataType<void> : std::integral_constant<DataType, DataType::kChar> {};

template <>
struct GetDataType<float> : std ::integral_constant<DataType, DataType ::kFloat> {};
inline float GetTypeByDataType(std ::integral_constant<DataType, DataType ::kFloat>) { return {}; }
template <>
struct GetDataType<double> : std ::integral_constant<DataType, DataType ::kDouble> {};
inline double GetTypeByDataType(std ::integral_constant<DataType, DataType ::kDouble>) { return {}; }
template <>
struct GetDataType<int8_t> : std ::integral_constant<DataType, DataType ::kInt8> {};
inline int8_t GetTypeByDataType(std ::integral_constant<DataType, DataType ::kInt8>) { return {}; }
template <>
struct GetDataType<int32_t> : std ::integral_constant<DataType, DataType ::kInt32> {};
inline int32_t GetTypeByDataType(std ::integral_constant<DataType, DataType ::kInt32>) { return {}; }
template <>
struct GetDataType<int64_t> : std ::integral_constant<DataType, DataType ::kInt64> {};
inline int64_t GetTypeByDataType(std ::integral_constant<DataType, DataType ::kInt64>) { return {}; }
template <>
struct GetDataType<char> : std ::integral_constant<DataType, DataType ::kChar> {};
inline char GetTypeByDataType(std ::integral_constant<DataType, DataType ::kChar>) { return {}; }
template <>
struct GetDataType<uint8_t> : std ::integral_constant<DataType, DataType ::kUInt8> {};
inline uint8_t GetTypeByDataType(std ::integral_constant<DataType, DataType ::kUInt8>) { return {}; }
template <>
struct GetDataType<bool> : std ::integral_constant<DataType, DataType ::kBool> {};
inline bool GetTypeByDataType(std ::integral_constant<DataType, DataType ::kBool>) { return {}; }
// template <>
// struct GetDataType<OFRecord> : std ::integral_constant<DataType, DataType ::kOFRecord> {};
// inline OFRecord GetTypeByDataType(std ::integral_constant<DataType, DataType ::kOFRecord>) { return {}; }
template <>
struct GetDataType<uint32_t> : std ::integral_constant<DataType, DataType ::kUInt32> {};
inline uint32_t GetTypeByDataType(std ::integral_constant<DataType, DataType ::kUInt32>) { return {}; }
// template <>
// struct GetDataType<float16> : std ::integral_constant<DataType, DataType ::kFloat16> {};
// inline float16 GetTypeByDataType(std ::integral_constant<DataType, DataType ::kFloat16>) { return {}; }
// template <>
// struct GetDataType<bfloat16> : std ::integral_constant<DataType, DataType ::kBFloat16> {};
// inline bfloat16 GetTypeByDataType(std ::integral_constant<DataType, DataType ::kBFloat16>) { return {}; }
// template <>
// struct GetDataType<std ::complex<float> > : std ::integral_constant<DataType, DataType ::kComplex64> {};
// inline std ::complex<float> GetTypeByDataType(std ::integral_constant<DataType, DataType ::kComplex64>) { return {};
// } template <> struct GetDataType<std ::complex<double> > : std ::integral_constant<DataType, DataType ::kComplex128>
// {}; inline std ::complex<double> GetTypeByDataType(std ::integral_constant<DataType, DataType ::kComplex128>) {
// return {}; }
template <>
struct GetDataType<uint64_t> : std ::integral_constant<DataType, DataType ::kUInt64> {};
inline uint64_t GetTypeByDataType(std ::integral_constant<DataType, DataType ::kUInt64>) { return {}; }
template <>
struct GetDataType<int16_t> : std ::integral_constant<DataType, DataType ::kInt16> {};
inline int16_t GetTypeByDataType(std ::integral_constant<DataType, DataType ::kInt16>) { return {}; };
}  // namespace fineflow
#endif  // !fineflow_CORE_COMMON_DATA_TYPE_H_
