module;
#include "fineflow/core/common/device_type.pb.h"

export module fineflow.core.common.device_type_proto;
import std;
export namespace fineflow {
using DeviceType = DeviceType;
template <typename T>
inline auto DeviceTypeName(T value) {
  return DeviceType_Name(value);
}
}  // namespace fineflow
export namespace std {
template <>
struct formatter<fineflow::DeviceType> : formatter<string> {
  template <typename FormatContext>
  auto format(const fineflow::DeviceType type, FormatContext& ctx) const {
    return std::format_to(ctx.out(), "{}", fineflow::DeviceTypeName(type).substr(1));
  }
};
}  // namespace std
