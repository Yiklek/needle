export module fineflow.core.common.hash;
import std.compat;
export namespace fineflow {

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

template <typename Key, typename T, typename Hash = std::hash<Key>>
using HashMap = std::unordered_map<Key, T, Hash>;

template <typename Key, typename Hash = std::hash<Key>>
using HashSet = std::unordered_set<Key, Hash>;

}  // namespace fineflow
export namespace std {

template <typename T1, typename T2>
struct hash<std::pair<T1, T2>> {
  std::size_t operator()(const std::pair<T1, T2>& p) const { return fineflow::Hash(p.first, p.second); }
};
}  // namespace std
