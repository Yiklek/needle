#ifndef FINEFLOW_CORE_COMMON_TRAITS_HPP_
#define FINEFLOW_CORE_COMMON_TRAITS_HPP_
#include <type_traits>
namespace fineflow {
#if __cplusplus < 202002L
template <class T>
struct type_identity {
  using type = T;
};
#else
using std::type_identity;
#endif

template <bool B, class TrueType, class FalseType>
using IfElseT = std::conditional_t<B, TrueType, FalseType>;

}  // namespace fineflow
#endif  // FINEFLOW_CORE_COMMON_TRAITS_HPP_
