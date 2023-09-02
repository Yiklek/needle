#ifndef FINEFLOW_CORE_COMMON_AUTO_REGISTER_HPP_
#define FINEFLOW_CORE_COMMON_AUTO_REGISTER_HPP_

#include <functional>
#include <iostream>
#include <memory>

#include "fineflow/core/common/hash_container.h"
#include "fineflow/core/common/log.h"
namespace fineflow {

template <typename Key, typename Base, typename... Args>
struct AutoRegistrationFactory {
public:
  using Creator = std::function<Base*(Args&&...)>;
  template <typename Derived>
  struct RawRegisterType {
    RawRegisterType(Key k) {  // NOLINT
      AutoRegistrationFactory<Key, Base, Args...>::Get().creatorsMut()->emplace(
          k, [](Args&&... args) { return new Derived(std::forward<Args>(args)...); });
    }
  };

  struct CreatorRegisterType {
    CreatorRegisterType(Key k, Creator v) {
      AutoRegistrationFactory<Key, Base, Args...>::Get().creatorsMut()->emplace(k, v);
    }
  };

  Base* New(Key k, Args&&... args) const {
    auto creators_it = creators().find(k);
    if (creators_it == creators().end()) {
      LOG(err) << "Unregistered: key: " << k << "  Base type name:" << typeid(Base).name() << "  Key type name"
               << typeid(Key).name();
    }
    return creators_it->second(std::forward<Args>(args)...);
  }

  bool isClassRegistered(Key k, Args&&... /*args*/) const { return creators().find(k) != creators().end(); }

  static AutoRegistrationFactory<Key, Base, Args...>& Get() {
    static AutoRegistrationFactory<Key, Base, Args...> obj;
    return obj;
  }

private:
  std::unique_ptr<HashMap<Key, Creator>> creators_;

  [[nodiscard]] bool hasCreators() const { return creators_.get() != nullptr; }

  const HashMap<Key, Creator>& creators() const {
    if (!hasCreators()) {
      LOG(err) << "Unregistered key type: " << typeid(Key).name();
    }
    return *creators_.get();
  }

  HashMap<Key, Creator>* creatorsMut() {
    if (!creators_) {
      creators_.reset(new HashMap<Key, Creator>);
    }
    return creators_.get();
  }
};

#define REGISTER_VAR_NAME FF_PP_CAT(g_registry_var, __COUNTER__)

#define REGISTER_CLASS(Key, k, Base, Derived) \
  static AutoRegistrationFactory<Key, Base>::RawRegisterType<Derived> REGISTER_VAR_NAME(k)
#define REGISTER_CLASS_WITH_ARGS(Key, k, Base, Derived, ...) \
  static AutoRegistrationFactory<Key, Base, __VA_ARGS__>::RawRegisterType<Derived> REGISTER_VAR_NAME(k)
#define REGISTER_CLASS_CREATOR(Key, k, Base, f, ...) \
  static AutoRegistrationFactory<Key, Base, ##__VA_ARGS__>::CreatorRegisterType REGISTER_VAR_NAME(k, f)

template <typename Key, typename Base, typename... Args>
inline Base* NewObj(Key k, Args&&... args) {
  return AutoRegistrationFactory<Key, Base, Args...>::Get().New(k, std::forward<Args>(args)...);
}

template <typename Key, typename Base, typename... Args>
inline std::unique_ptr<Base> NewObjUniquePtr(Key k, Args&&... args) {
  return std::unique_ptr<Base>(AutoRegistrationFactory<Key, Base, Args...>::Get().New(k, std::forward<Args>(args)...));
}

template <typename Key, typename Base, typename... Args>
inline bool IsClassRegistered(Key k, Args&&... args) {
  return AutoRegistrationFactory<Key, Base, Args...>::Get().IsClassRegistered(k, std::forward<Args>(args)...);
}
}  // namespace fineflow
#endif  // FINEFLOW_CORE_COMMON_AUTO_REGISTER_HPP_
