#ifndef FINEFLOW_CORE_COMMON_REGISTER_MANAGER_HPP_
#define FINEFLOW_CORE_COMMON_REGISTER_MANAGER_HPP_
#include "preprocess.h"
#define REGISTER_VAR_NAME_IMPL(line, counter) FF_PP_CONCAT_WITH_3(_, t, line, counter)
#define REGISTER_VAR_NAME REGISTER_VAR_NAME_IMPL(__LINE__, __COUNTER__)

#define REGISTER_KEY_WITH_CLASS(class_key, class_value, key) \
  static RegisterTrigger<class_key, class_value> REGISTER_VAR_NAME = Registry<class_key, class_value>((key))

#define REGISTER_KEY_WITH_CLASS_T(class_key, class_value, class_type, key) \
  static RegisterTrigger<class_key, class_value, class_type> REGISTER_VAR_NAME = Registry<class_key, class_value>((key))

#define REGISTER_KEY(class_value, key) REGISTER_KEY_WITH_CLASS(decltype((key)), class_value, key)
#define REGISTER_KEY_VALUE(key, value) \
  REGISTER_KEY_WITH_CLASS(decltype((key)), decltype((value)), key).setValue((value))

#define REGISTER_KEY_VALUE_T(class_type, key, value) \
  REGISTER_KEY_WITH_CLASS_T(decltype((key)), decltype((value)), class_type, key).setValue((value))

#define REGISTER_KEY_VALUE_MGR(mgr, key, value) \
  REGISTER_KEY_WITH_CLASS_T(mgr::Key, mgr::Value, mgr::Tag, key).setValue((value))

#endif  // FINEFLOW_CORE_COMMON_REGISTER_MANAGER_HPP_
