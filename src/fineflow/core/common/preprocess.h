#ifndef FINEFLOW_CORE_COMMON_PREPROCESS_H_
#define FINEFLOW_CORE_COMMON_PREPROCESS_H_
#include "boost/preprocessor.hpp"
#define FF_PP_STRINGIZE(...) FF_PP_STRINGIZE_I(__VA_ARGS__)
#define FF_PP_STRINGIZE_I(...) #__VA_ARGS__

#define FF_PP_CAT(a, b) FF_PP_CAT_I(a, b)
#define FF_PP_CAT_I(a, b) a##b

#define FF_PP_TUPLE_APPEND(tuple, elem) BOOST_PP_VARIADIC_TO_TUPLE(BOOST_PP_TUPLE_ENUM(tuple), elem)
#define FF_PP_TUPLE_PREPEND(tuple, elem) BOOST_PP_VARIADIC_TO_TUPLE(elem, BOOST_PP_TUPLE_ENUM(tuple))

#define FF_PP_FORWARD(macro, ...) macro(__VA_ARGS__)

#define FF_PP_ALL(...) __VA_ARGS__

#define FF_PP_JOIN_RESET "fineflow/core/common/pp/join_reset.h"

#include FF_PP_JOIN_RESET
#define FF_PP_JOIN_I(s, state, x) BOOST_PP_CAT(state, BOOST_PP_CAT(FF_PP_JOIN_SEP, x))
#define FF_PP_JOIN(...)                                                                          \
  BOOST_PP_SEQ_FOLD_LEFT(FF_PP_JOIN_I, BOOST_PP_SEQ_HEAD(BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)), \
                         BOOST_PP_SEQ_TAIL(BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)))

#endif  // FINEFLOW_CORE_COMMON_PREPROCESS_H_
