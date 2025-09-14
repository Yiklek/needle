module;

#include "fineflow/core/common/result.h"
export module fineflow.core.common.registry_manager;
import std;
import fineflow.core.common.data_type_proto;
import fineflow.core.common.hash;
import fineflow.core.common.error;
import fineflow.core.common.log;

export namespace fineflow {

template <class Key, class Value, class Type = void>
class RegistryMgr {
private:
  RegistryMgr() = default;

public:
  RegistryMgr(RegistryMgr const&) = delete;
  RegistryMgr& operator=(RegistryMgr const&) = delete;
  static RegistryMgr& Get() {
    static RegistryMgr<Key, Value, Type> mgr;
    return mgr;
  }

  Ret<void> Register(Key&& key, Value&& value) {
    // std::cout << "Register: " << key << std::endl;
    // std::cerr << "Register: key:" << key << " " << FF_PP_STACK_FUNC << std::endl;
    // int i = 0;
    CHECK_OR_RETURN(result_.emplace(std::forward<Key>(key), std::forward<Value>(value)).second);
    // << "Register key: " << key << " failed.";
    return {};
  }
  Ret<const Value* const> GetValue(const Key& key) {
    // std::cerr << "GetValue: key:" << key << " " << FF_PP_STACK_FUNC << std::endl;
    auto it = result_.find(key);
    CHECK_OR_RETURN(it != result_.end()) << "RegistryMgr Value for key:(" << key << ") not found. ";
    return &(it->second);
  }
  bool IsRegistered(const Key& key) { return result_.count(key) != 0; }

  const HashMap<Key, Value>& GetAll() { return result_; };

private:
  HashMap<Key, Value> result_;
};

template <class Key, class Value>
struct Registry {
protected:
  Key key_;
  Value value_;

public:
  explicit Registry(const Key& key) : key_(key) {}
  Key& key() { return key_; }
  Value& finish() { return value_; }

  Registry<Key, Value>&& setValue(Value&& value) && {
    value_ = std::forward<Value>(value);
    return std::move(*this);
  }
};

template <class Key, class Value, class Type = void>
struct RegisterTrigger final {
  RegisterTrigger(const Registry<Key, Value>& registry) {  // NOLINT
    std::cout << "RegisterTrigger: " << registry.key() << std::endl;
    (void)RegistryMgr<Key, Value, Type>::Get().Register(registry.key(), registry.finish());
  }
  RegisterTrigger(Registry<Key, Value>&& registry) {  // NOLINT
    (void)RegistryMgr<Key, Value, Type>::Get().Register(std::move(registry.key()), std::move(registry.finish()));
  }
};

/**
 * @brief Multiple registry helper
 *
 * Example:
 * auto r = Register<>::New(1, 2, 3, 4)
 * registers RegistryMgr<int, int, void> as
 * 1 -> 2, 3 -> 4
 */
template <class Type = void>
struct Register {
  static constexpr std::size_t GroupSize = 2;
  template <class... Args>
  struct Impl {
    static_assert(sizeof...(Args) % GroupSize == 0);
    template <std::size_t I>
    using ArgType = typename std::tuple_element<I, std::tuple<Args...>>::type;

    explicit Impl(Args&&... args) {
      auto indexes2 = std::make_index_sequence<sizeof...(Args) / GroupSize>();
      RegisterAll(args..., indexes2);
    }

    template <std::size_t... GroupI>
    inline void RegisterAll(Args... args, std::index_sequence<GroupI...>) {
      (RegisterGroup<GroupI>(std::tuple(args...), std::make_index_sequence<GroupSize>()), ...);
    }
    template <std::size_t Group, std::size_t... I>
    inline void RegisterGroup(std::tuple<Args...> args, std::index_sequence<I...>) {
      RegistryMgr<ArgType<GroupSize * Group + I>..., Type>::Get().Register(
          std::move(std::get<GroupSize * Group + I>(args))...);
    }
  };
  Register() = delete;
  template <class... Args>
  static inline bool New(Args&&... args) {
    Impl<Args...>(std::forward<Args>(args)...);
    return true;
  }
};

using RegisterVoid = Register<>;

#define REGISTER_VAR_NAME FF_PP_JOIN_U(t, __LINE__, __COUNTER__)

#define REGISTER_KEY_WITH_CLASS(class_key, class_value, key) \
  static RegisterTrigger<class_key, class_value> REGISTER_VAR_NAME = Registry<class_key, class_value>((key))

#define REGISTER_KEY_WITH_CLASS_T(class_key, class_value, class_type, key) \
  static RegisterTrigger<class_key, class_value, class_type> REGISTER_VAR_NAME = Registry<class_key, class_value>((key))

#define REGISTER_KEY(class_value, key) REGISTER_KEY_WITH_CLASS(decltype((key)), class_value, key)
#define REGISTER_KEY_VALUE(key, value) \
  REGISTER_KEY_WITH_CLASS(decltype((key)), decltype((value)), key).setValue((value))

#define REGISTER_KEY_VALUE_T(class_type, key, value) \
  REGISTER_KEY_WITH_CLASS_T(decltype((key)), decltype((value)), class_type, key).setValue((value))

}  // namespace fineflow
