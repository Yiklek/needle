#ifndef FINEFLOW_CORE_COMMON_FUNCTION_TRAITS_HPP_
#define FINEFLOW_CORE_COMMON_FUNCTION_TRAITS_HPP_
#include <tuple>
namespace fineflow {

template <typename... Args>
using void_t = void;

template <typename T, typename = void>
struct function_traits;

template <typename Ret, typename... Args>
struct function_traits<Ret(Args...)> {
  using func_type = Ret(Args...);
  using return_type = Ret;
  using args_type = std::tuple<Args...>;
  template <std::size_t I>
  using arg_type = typename std::tuple_element<I, args_type>::type;

  static constexpr std::size_t Nargs = sizeof...(Args);
};

template <typename Ret, typename... Args>
struct function_traits<Ret (*)(Args...)> {
  using func_type = Ret(Args...);
  using return_type = Ret;
  using args_type = std::tuple<Args...>;
  template <std::size_t I>
  using arg_type = typename std::tuple_element<I, args_type>::type;

  static constexpr std::size_t Nargs = sizeof...(Args);
};

template <typename Ret, typename C, typename... Args>
struct function_traits<Ret (C::*)(Args...)> {
  using func_type = Ret(Args...);
  using return_type = Ret;
  using args_type = std::tuple<Args...>;
  template <std::size_t I>
  using arg_type = typename std::tuple_element<I, args_type>::type;

  static constexpr std::size_t Nargs = sizeof...(Args);
};

template <typename Ret, typename C, typename... Args>
struct function_traits<Ret (C::*)(Args...) const> {
  using func_type = Ret(Args...);
  using return_type = Ret;
  using args_type = std::tuple<Args...>;
  template <std::size_t I>
  using arg_type = typename std::tuple_element<I, args_type>::type;

  static constexpr std::size_t Nargs = sizeof...(Args);
};

template <typename F>
struct function_traits<F, void_t<decltype(&F::operator())>> : public function_traits<decltype(&F::operator())> {};

template <typename F>
using FuncType = function_traits<F>::func_type;

}  // namespace fineflow
#endif  // !fineflow_CORE_COMMON_FUNCTION_TRAITS_HPP_
