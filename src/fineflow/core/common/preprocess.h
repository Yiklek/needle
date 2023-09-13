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

#define FF_PP_JOIN_STATE_RET(state) BOOST_PP_TUPLE_ELEM(0, state)
#define FF_PP_JOIN_STATE_SEP(state) BOOST_PP_TUPLE_ELEM(1, state)
#define FF_PP_JOIN_NEXT_RET(state, x) \
  BOOST_PP_CAT(FF_PP_JOIN_STATE_RET(state), BOOST_PP_CAT(FF_PP_JOIN_STATE_SEP(state), x))
#define FF_PP_JOIN_STATE_NEXT_TUPLE(state, x) \
  BOOST_PP_VARIADIC_TO_TUPLE(FF_PP_JOIN_NEXT_RET(state, x), FF_PP_JOIN_STATE_SEP(state))
#define FF_PP_JOIN_START_TUPLE(sep, start, ...) BOOST_PP_VARIADIC_TO_TUPLE(start, sep)

#define FF_PP_JOIN_I(s, state, x) FF_PP_JOIN_STATE_NEXT_TUPLE(state, x)

#define FF_PP_JOIN_SEQ(sep, SEQ)                                                                                 \
  FF_PP_JOIN_STATE_RET(BOOST_PP_SEQ_FOLD_LEFT(FF_PP_JOIN_I, FF_PP_JOIN_START_TUPLE(sep, BOOST_PP_SEQ_HEAD(SEQ)), \
                                              BOOST_PP_SEQ_TAIL(SEQ)))
#define FF_PP_JOIN(sep, ...) FF_PP_JOIN_SEQ(sep, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define FF_PP_JOIN_U(...) FF_PP_JOIN(_, __VA_ARGS__)

#endif  // FINEFLOW_CORE_COMMON_PREPROCESS_H_
