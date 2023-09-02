#ifndef FINEFLOW_CORE_COMMON_RESULT_TYPES_H_
#define FINEFLOW_CORE_COMMON_RESULT_TYPES_H_
#include "tl/expected.hpp"
namespace fineflow {

template <class T, class E>
using Maybe = tl::expected<T, E>;

template <class E>
using Failure = tl::unexpected<E>;

}  // namespace fineflow
#endif  // FINEFLOW_CORE_COMMON_RESULT_TYPES_H_
