export module fineflow.core.common.util;
export namespace fineflow {

template <class... Args>
constexpr bool Or(Args... args) {
  return (args || ...);
}

template <class... Args>
constexpr bool And(Args... args) {
  return (args && ...);
}

}  // namespace fineflow
