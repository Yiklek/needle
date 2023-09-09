#ifndef FINEFLOW_CORE_COMMON_FMT_HPP_
#define FINEFLOW_CORE_COMMON_FMT_HPP_
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#include "fmt/format.h"
#include "fmt/ostream.h"
#include "fmt/ranges.h"
#include "fmt/std.h"

namespace fineflow {
template <class T>
inline constexpr bool IsStreamableV =
    !(std::is_arithmetic_v<T> || std::is_array_v<T> || std::is_pointer_v<T> || std::is_same_v<T, char> ||
      std::is_convertible_v<T, std::string_view> || std::is_convertible_v<T, std::string> ||
      std::is_same_v<T, std::string_view> || (std::is_convertible_v<T, int> && !std::is_enum_v<T>));

template <typename T>
using ostreamable_t = std::enable_if_t<fmt::has_formatter<T, fmt::format_context>::value && IsStreamableV<T>>;
template <typename T>
using formatable_t = std::enable_if_t<fmt::has_formatter<T, fmt::format_context>::value>;

#if __cplusplus >= 202002L
template <typename T>
concept formatable = fmt::has_formatter<T, fmt::format_context>::value;

// clang-format off
template <typename T>
concept ostreamable = formatable<T> && IsStreamableV<T>;
// clang-format on
#endif

}  // namespace fineflow

namespace fineflow {

// NOLINTBEGIN
template <typename T>
class fmt_unique {
  friend struct fmt::formatter<fmt_unique<T>>;
  const std::unique_ptr<T>& ptr;

public:
  explicit fmt_unique(const std::unique_ptr<T>& ptr) : ptr(ptr) {}
};
template <typename T>
class fmt_shared {
  friend struct fmt::formatter<fmt_shared<T>>;
  const std::shared_ptr<T>& ptr;

public:
  explicit fmt_shared(const std::shared_ptr<T>& ptr) : ptr(ptr) {}
};
template <typename T>
class fmt_weak {
  friend struct fmt::formatter<fmt_weak<T>>;
  const std::weak_ptr<T> ptr;

public:
  explicit fmt_weak(const std::weak_ptr<T>& ptr) : ptr(ptr) {}
};
struct ShortFormat;
}  // namespace fineflow

template <>
struct fmt::formatter<fineflow::ShortFormat> {
  // f: full
  // s: short
  char presentation = 'f';
  constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    const auto* it = ctx.begin();
    const auto* end = ctx.end();
    if (it != end && (*it == 's' || *it == 'f')) {
      presentation = *it++;
    }

    // Check if reached the end of the range:
    if (it != end && *it != '}') {
      throw format_error("invalid format");
    }

    // Return an iterator past the end of the parsed range:
    return it;
  }
};
template <typename T>
struct fmt::formatter<std::shared_ptr<T>> : public fmt::formatter<T> {
  template <typename FormatContext, class = fineflow::formatable_t<T>>
  auto format(const std::shared_ptr<T>& ptr, FormatContext& ctx) const -> decltype(ctx.out()) {
    return fmt::formatter<T>::format(*ptr.get(), ctx);
  }
};
template <typename T>
struct fmt::formatter<fineflow::fmt_shared<T>> : public fmt::formatter<T> {
  template <typename FormatContext, class = fineflow::formatable_t<T>>
  auto format(const fineflow::fmt_shared<T>& proxy, FormatContext& ctx) const -> decltype(ctx.out()) {
    return fmt::formatter<T>::format(*proxy.ptr, ctx);
  }
};
template <typename T>
struct fmt::formatter<fineflow::fmt_weak<T>> : public fmt::formatter<fineflow::fmt_shared<T>> {
  template <typename FormatContext, class = fineflow::formatable_t<T>>
  auto format(const fineflow::fmt_weak<T>& proxy, FormatContext& ctx) const -> decltype(ctx.out()) {
    return fmt::formatter<fineflow::fmt_shared<T>>::format(fmt_shared(proxy.ptr.lock()), ctx);
  }
};
template <typename T>
struct fmt::formatter<fineflow::fmt_unique<T>> : public fmt::formatter<T> {
  template <typename FormatContext, class = fineflow::formatable_t<T>>
  auto format(const fineflow::fmt_unique<T>& proxy, FormatContext& ctx) const -> decltype(ctx.out()) {
    return fmt::formatter<T>::format(*proxy.ptr, ctx);
  }
};

namespace std {
#if __cplusplus >= 202002L
template <fineflow::ostreamable T>
#else
template <typename T, fineflow::ostreamable_t<T> = 0>
#endif
inline std::ostream& operator<<(std::ostream& os, const T& t) {
  os << fmt::to_string(t);
  return os;
}

#if __cplusplus >= 202002L
template <fineflow::ostreamable T>
#else
template <typename T, class = fineflow::ostreamable_t<T>>
#endif
inline std::ostringstream& operator<<(std::ostringstream& os, const T& t) {
  os << fmt::to_string(t);
  return os;
}
}  // namespace std
// NOLINTEND
#endif  // FINEFLOW_CORE_COMMON_FMT_HPP_
