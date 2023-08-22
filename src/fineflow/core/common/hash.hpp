#ifndef FINEFLOW_CORE_CORE_COMMON_HASH_HPP_
#define FINEFLOW_CORE_CORE_COMMON_HASH_HPP_

#include <cstddef>
#include <functional>
#include <string>
namespace fineflow {

inline size_t HashCombine(size_t lhs, size_t rhs) { return lhs ^ (rhs + 0x9e3779b9 + (lhs << 6U) + (lhs >> 2U)); }

inline void HashCombine(size_t* seed, size_t hash) { *seed = HashCombine(*seed, hash); }

template <typename... T>
inline void AddHash(size_t* seed, const T&... v) {
  (HashCombine(seed, std::hash<T>()(v)), ...);
}

template <typename T, typename... Ts>
inline size_t Hash(const T& v1, const Ts&... vn) {
  size_t seed = std::hash<T>()(v1);

  AddHash<Ts...>(&seed, vn...);

  return seed;
}
}  // namespace fineflow
namespace std {

template <>
struct hash<std::pair<std::string, int32_t>> {
  std::size_t operator()(const std::pair<std::string, int32_t>& p) const { return fineflow::Hash(p.first, p.second); }
};
}  // namespace std
#endif
