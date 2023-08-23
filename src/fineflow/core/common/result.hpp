#ifndef FINEFLOW_CORE_COMMON_RESULT_HPP_
#define FINEFLOW_CORE_COMMON_RESULT_HPP_
#include "fineflow/core/common/error.h"
#include "fineflow/core/common/preprocess.h"
#include "tl/expected.hpp"
namespace fineflow {
template <class T, class E = Error>
using Ret = tl::expected<T, E>;

template <class E = Error>
inline tl::unexpected<E> Fail(E&& e) {
  return tl::unexpected(std::forward<E>(e));
}

template <class T>
inline Ret<T> Ok() {
  return {};
}

#define CHECK_OR_RETURN_INTERNAL(expr, error_msg)                                              \
  if (!(expr))                                                                                 \
  return Error::CheckFailedError().addStackFrame([](const char* function) {                    \
    thread_local static auto frame = ErrorStackFrame(__FILE__, __LINE__, function, error_msg); \
    return frame;                                                                              \
  }(__FUNCTION__))

#define CHECK_OR_RETURN(expr)                                            \
  CHECK_OR_RETURN_INTERNAL(expr, FF_PP_STRINGIZE(CHECK_OR_RETURN(expr))) \
      << "Check failed: (" << FF_PP_STRINGIZE(expr) << ") " << Error::kOverrideThenMergeMessage

#define CHECK_EQ_OR_RETURN(lhs, rhs)                                                      \
  CHECK_OR_RETURN_INTERNAL((lhs) == (rhs), FF_PP_STRINGIZE(CHECK_EQ_OR_RETURN(lhs, rhs))) \
      << "Check failed: (" << (lhs) << " == " << (rhs) << ") " << Error::kOverrideThenMergeMessage

#define CHECK_GE_OR_RETURN(lhs, rhs)                                                      \
  CHECK_OR_RETURN_INTERNAL((lhs) >= (rhs), FF_PP_STRINGIZE(CHECK_GE_OR_RETURN(lhs, rhs))) \
      << "Check failed: (" << (lhs) << " >= " << (rhs) << ") " << Error::kOverrideThenMergeMessage

#define CHECK_GT_OR_RETURN(lhs, rhs)                                                     \
  CHECK_OR_RETURN_INTERNAL((lhs) > (rhs), FF_PP_STRINGIZE(CHECK_GT_OR_RETURN(lhs, rhs))) \
      << "Check failed: (" << (lhs) << " > " << (rhs) << ") " << Error::kOverrideThenMergeMessage

#define CHECK_LE_OR_RETURN(lhs, rhs)                                                      \
  CHECK_OR_RETURN_INTERNAL((lhs) <= (rhs), FF_PP_STRINGIZE(CHECK_LE_OR_RETURN(lhs, rhs))) \
      << "Check failed: (" << (lhs) << " <= " << (rhs) << ") " << Error::kOverrideThenMergeMessage

#define CHECK_LT_OR_RETURN(lhs, rhs)                                                     \
  CHECK_OR_RETURN_INTERNAL((lhs) < (rhs), FF_PP_STRINGIZE(CHECK_LT_OR_RETURN(lhs, rhs))) \
      << "Check failed: (" << (lhs) << " < " << (rhs) << ") " << Error::kOverrideThenMergeMessage

#define CHECK_NE_OR_RETURN(lhs, rhs)                                                      \
  CHECK_OR_RETURN_INTERNAL((lhs) != (rhs), FF_PP_STRINGIZE(CHECK_NE_OR_RETURN(lhs, rhs))) \
      << "Check failed: (" << (lhs) << " != " << (rhs) << ") " << Error::kOverrideThenMergeMessage

}  // namespace fineflow
#endif
