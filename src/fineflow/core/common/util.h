#ifndef FINEFLOW_CORE_COMMON_UTIL_H_
#define FINEFLOW_CORE_COMMON_UTIL_H_
namespace fineflow {
template <class... Args>
constexpr bool Or(Args... args) {
  return (args || ...);
}

template <class... Args>
constexpr bool And(Args... args) {
  return (args && ...);
}

#define FF_DISALLOW_COPY(ClassName)     \
  /* NOLINTNEXTLINE */                  \
  ClassName(const ClassName&) = delete; \
  /* NOLINTNEXTLINE */                  \
  ClassName& operator=(const ClassName&) = delete;

#define FF_DISALLOW_MOVE(ClassName) \
  /* NOLINTNEXTLINE */              \
  ClassName(ClassName&&) = delete;  \
  /* NOLINTNEXTLINE */              \
  ClassName& operator=(ClassName&&) = delete;

#define FF_DISALLOW_COPY_AND_MOVE(ClassName) \
  FF_DISALLOW_COPY(ClassName)                \
  FF_DISALLOW_MOVE(ClassName)

#define FF_DEFAULT_COPY(ClassName) \
  /* NOLINTNEXTLINE */             \
  ClassName(const ClassName&) = default;
#define FF_DEFAULT_MOVE(ClassName) \
  /* NOLINTNEXTLINE */             \
  ClassName(ClassName&&) = default;

#define FF_DEFAULT_COPY_AND_MOVE(ClassName) \
  FF_DEFAULT_COPY(ClassName)                \
  FF_DEFAULT_MOVE(ClassName)
}  // namespace fineflow
#endif
