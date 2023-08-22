#ifndef FINEFLOW_CORE_COMMON_HASH_CONTAINER_
#define FINEFLOW_CORE_COMMON_HASH_CONTAINER_

#include <unordered_map>
#include <unordered_set>

namespace fineflow {

template <typename Key, typename T, typename Hash = std::hash<Key>>
using HashMap = std::unordered_map<Key, T, Hash>;

template <typename Key, typename Hash = std::hash<Key>>
using HashSet = std::unordered_set<Key, Hash>;

}  // namespace fineflow

#endif  // fineflow_CORE_COMMON_HASH_CONTAINER_
