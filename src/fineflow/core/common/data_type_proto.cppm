module;
#include "fineflow/core/common/data_type.pb.h"

export module fineflow.core.common.data_type_proto;
import std;
export namespace fineflow {
using DataType = DataType;
template <typename T>
inline auto DataTypeName(T value) {
  return DataType_Name(value);
}
}  // namespace fineflow
export namespace std {
template <>
struct formatter<fineflow::DataType> : formatter<string> {
  template <typename FormatContext>
  auto format(const fineflow::DataType dtype, FormatContext& ctx) const {
    return std::format_to(ctx.out(), "{}", fineflow::DataTypeName(dtype).substr(1));
  }
};
}  // namespace std
