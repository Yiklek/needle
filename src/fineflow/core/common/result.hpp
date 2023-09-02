#ifndef FINEFLOW_CORE_COMMON_RESULT_HPP_
#define FINEFLOW_CORE_COMMON_RESULT_HPP_
#include "fineflow/core/common/error.h"
#include "fineflow/core/common/preprocess.h"
#include "fineflow/core/common/result_types.h"

namespace fineflow {
template <class T, class E = Error>
using Ret = Maybe<T, E>;

template <class E = Error>
inline Failure<E> Fail(E&& e) {
  return Failure<E>(std::forward<E>(e));
}

struct Ok {
  template <class T>
  inline operator Ret<T>() {
    return {};
  }
};
#define RET_ERROR_ADD_STACKFRAME(error, error_msg)                                                         \
  (error).addStackFrame([](const char* function) {                                                         \
    thread_local static auto frame = ::fineflow::ErrorStackFrame(__FILE__, __LINE__, function, error_msg); \
    return frame;                                                                                          \
  }(__FUNCTION__))

#define CHECK_OR_RETURN_INTERNAL(expr, error_msg) \
  if (!(expr)) return RET_ERROR_ADD_STACKFRAME(Error::CheckFailedError(), error_msg)

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

#define TRY_ASSIGN_IMPL(result, lhs, rexpr)                                                                \
  /*NOLINTNEXTLINE*/                                                                                       \
  auto result = (rexpr);                                                                                   \
  if (!(result).has_value()) {                                                                             \
    return RET_ERROR_ADD_STACKFRAME(std::move((result)).error(), FF_PP_STRINGIZE(TRY_ASSIGN(lhs, rexpr))); \
  }                                                                                                        \
  /*NOLINTNEXTLINE*/                                                                                       \
  lhs = *std::move((result))

#define TRY_ASSIGN(lhs, rexpr) TRY_ASSIGN_IMPL(FF_PP_CAT(_result_or_value, __COUNTER__), lhs, rexpr)

#define TRY_CATCH_IMPL(result, rexpr, catch_exprs, stack_error_msg)       \
  /*NOLINTNEXTLINE*/                                                      \
  auto result = (rexpr);                                                  \
  if (!(result).has_value()) {                                            \
    auto e = RET_ERROR_ADD_STACKFRAME((result).error(), stack_error_msg); \
    catch_exprs                                                           \
  }

#define TRY_CATCH(rexpr, catch_exprs)                                                                \
  TRY_CATCH_IMPL(FF_PP_CAT(_result_or_value, __COUNTER__), FF_PP_ALL(rexpr), FF_PP_ALL(catch_exprs), \
                 FF_PP_STRINGIZE(TRY_CATCH(rexpr, ...)))

#define TRY_IMPL(result, rexpr) \
  TRY_CATCH_IMPL(               \
      result, rexpr, { return e; }, FF_PP_STRINGIZE(TRY(rexpr)))

#define TRY(rexpr) \
  TRY_CATCH_IMPL(  \
      FF_PP_CAT(_result_or_value, __COUNTER__), rexpr, { return e; }, FF_PP_STRINGIZE(TRY(rexpr)))

}  // namespace fineflow
#endif