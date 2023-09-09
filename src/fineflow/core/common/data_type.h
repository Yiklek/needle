#ifndef FINEFLOW_CORE_COMMON_DATA_TYPE_H_
#define FINEFLOW_CORE_COMMON_DATA_TYPE_H_
#include <complex>
#include <type_traits>

#include "fineflow/core/common/data_type.pb.h"
#include "fineflow/core/common/registry_manager.hpp"
#include "fineflow/core/common/traits.hpp"
#include "fineflow/core/common/types_tuple.h"
#include "fineflow/core/common/fmt.hpp"
namespace fineflow {

template <typename T, typename T2 = void>
struct GetDataType;

template <DataType Type>
struct DataTypeToClass;

template <>
struct GetDataType<void> : std::integral_constant<DataType, DataType::kChar> {};

template <DataType Type>
using DataTypeToType = typename DataTypeToClass<Type>::type;

using DataTypeSizeRegistryMgr = RegistryMgr<DataType, size_t>;

#define SPECIALIZE_GET_DATA_TYPE(type_cpp, type_proto)                          \
  REGISTER_KEY_VALUE(type_proto, sizeof(type_cpp));                             \
  template <>                                                                   \
  struct GetDataType<type_cpp> : std::integral_constant<DataType, type_proto> { \
    static constexpr size_t size = sizeof(type_cpp);                            \
  };                                                                            \
  template <>                                                                   \
  struct DataTypeToClass<type_proto> : type_identity<type_cpp> {                \
    static constexpr size_t size = sizeof(type_cpp);                            \
  };

#define FOR_SPECIALIZE_GET_DATA_TYPE(i, _, elem) FF_PP_FORWARD(SPECIALIZE_GET_DATA_TYPE, BOOST_PP_TUPLE_ENUM(elem))

BOOST_PP_SEQ_FOR_EACH(FOR_SPECIALIZE_GET_DATA_TYPE, _, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)
#undef FOR_SPECIALIZE_GET_DATA_TYPE

}  // namespace fineflow
template <>
struct fmt::formatter<fineflow::DataType> {
  static constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return ctx.end(); }
  template <typename FormatContext>
  auto format(const fineflow::DataType dtype, FormatContext& ctx) const -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(), "{}", fineflow::DataType_Name(dtype).substr(1));
  }
};
#endif  // fineflow_CORE_COMMON_DATA_TYPE_H_
