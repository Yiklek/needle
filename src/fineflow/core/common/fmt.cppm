export module fineflow.core.common.fmt;
import std;

#define FMT std
export namespace std {
template <typename T>
const void* ptr(T t)  // NOLINT
  requires(is_pointer_v<T> || is_same_v<T, nullptr_t>)
{
  return static_cast<void*>(t);
}
template <typename T>
std::string to_string(T&& t) {
  return std::format("{}", std::forward<T>(t));
  // return "";
}
}  // namespace std
export namespace fineflow {

template <typename T, typename Char = char>
using formatter = FMT::formatter<T>;
using format_context = FMT::format_context;
using FMT::format;
using FMT::format_to;
template <class T>
inline constexpr bool IsStreamableV =
    !(std::is_arithmetic_v<T> || std::is_array_v<T> || std::is_pointer_v<T> || std::is_same_v<T, char> ||
      std::is_convertible_v<T, std::string_view> || std::is_convertible_v<T, std::string> ||
      std::is_same_v<T, std::string_view> || (std::is_convertible_v<T, int> && !std::is_enum_v<T>));

template <typename T, typename Char>
using has_formatter = std::is_constructible<formatter<T, Char>>;

template <typename T>
using ostreamable_t = std::enable_if_t<has_formatter<T, format_context>::value && IsStreamableV<T>>;
template <typename T>
using formatable_t = std::enable_if_t<has_formatter<T, format_context>::value>;

#if __cplusplus >= 202002L
template <typename T>
concept formatable = has_formatter<T, format_context>::value;

// clang-format off
template <typename T>
concept ostreamable = formatable<T> && IsStreamableV<T>;
// clang-format on
#endif

}  // namespace fineflow
//
export namespace fineflow {

// NOLINTBEGIN
template <typename T>
class fmt_unique {
  friend struct FMT::formatter<fmt_unique<T>>;
  const std::unique_ptr<T>& ptr;

public:
  explicit fmt_unique(const std::unique_ptr<T>& ptr) : ptr(ptr) {}
};
template <typename T>
class fmt_shared {
  friend struct FMT::formatter<fmt_shared<T>>;
  const std::shared_ptr<T>& ptr;

public:
  explicit fmt_shared(const std::shared_ptr<T>& ptr) : ptr(ptr) {}
};
template <typename T>
class fmt_weak {
  friend struct FMT::formatter<fmt_weak<T>>;
  const std::weak_ptr<T> ptr;

public:
  explicit fmt_weak(const std::weak_ptr<T>& ptr) : ptr(ptr) {}
};
struct ShortFormat;
}  // namespace fineflow

export namespace std {
template <>
struct formatter<fineflow::ShortFormat> {
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
struct formatter<std::shared_ptr<T>> : public formatter<T> {
  template <typename FormatContext, class = fineflow::formatable_t<T>>
  auto format(const std::shared_ptr<T>& p, FormatContext& ctx) const -> decltype(ctx.out()) {
    return formatter<T>::format(p ? *p : ptr(nullptr), ctx);
  }
};
template <typename T>
struct formatter<fineflow::fmt_shared<T>> : public formatter<T> {
  template <typename FormatContext, class = fineflow::formatable_t<T>>
  auto format(const fineflow::fmt_shared<T>& proxy, FormatContext& ctx) const -> decltype(ctx.out()) {
    return formatter<T>::format(proxy.ptr ? *proxy.ptr : ptr(nullptr), ctx);
  }
};
template <typename T>
struct formatter<fineflow::fmt_weak<T>> : public formatter<fineflow::fmt_shared<T>> {
  template <typename FormatContext, class = fineflow::formatable_t<T>>
  auto format(const fineflow::fmt_weak<T>& proxy, FormatContext& ctx) const -> decltype(ctx.out()) {
    return formatter<fineflow::fmt_shared<T>>::format(fmt_shared(proxy.ptr.lock()), ctx);
  }
};
template <typename T>
struct formatter<fineflow::fmt_unique<T>> : public formatter<T> {
  template <typename FormatContext, class = fineflow::formatable_t<T>>
  auto format(const fineflow::fmt_unique<T>& proxy, FormatContext& ctx) const -> decltype(ctx.out()) {
    return formatter<T>::format(proxy.ptr ? *proxy.ptr : ptr(nullptr), ctx);
  }
};
}  // namespace std
export namespace std {
#if __cplusplus >= 202002L
template <fineflow::ostreamable T>
#else
template <typename T, fineflow::ostreamable_t<T> = 0>
#endif
inline std::ostream& operator<<(std::ostream& os, const T& t) {
  os << FMT::to_string(t);
  return os;
}

#if __cplusplus >= 202002L
template <fineflow::ostreamable T>
#else
template <typename T, class = fineflow::ostreamable_t<T>>
#endif
inline std::ostringstream& operator<<(std::ostringstream& os, const T& t) {
  os << FMT::to_string(t);
  return os;
}
}  // namespace std
