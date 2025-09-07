#ifndef FINEFLOW_CORE_COMMON_RESULT_TYPES_H_
#define FINEFLOW_CORE_COMMON_RESULT_TYPES_H_

#if __cplusplus >= 202302L
#include <expected>
namespace fineflow {
template <typename T1, typename T2>
using expected = std::expected<T1, T2>;

template <typename E>
using unexpected = std::unexpected<E>;
}  // namespace fineflow
#else
#include "tl/expected.hpp"
namespace fineflow {
template <typename T1, typename T2>
using expected = tl::expected<T1, T2>;

template <typename E>
using unexpected = tl::unexpected<E>;
}  // namespace fineflow
#endif

namespace fineflow {

template <class T, class E>
using Maybe = expected<T, E>;

template <class E>
using Failure = unexpected<E>;

}  // namespace fineflow
#endif  // FINEFLOW_CORE_COMMON_RESULT_TYPES_H_
