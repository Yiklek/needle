#ifndef FINEFLOW_CORE_COMMON_UTIL_H_
#define FINEFLOW_CORE_COMMON_UTIL_H_
namespace fineflow {
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

}  // namespace fineflow
#endif
