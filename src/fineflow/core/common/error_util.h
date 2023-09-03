#ifndef FINEFLOW_CORE_COMMON_ERROR_UTIL_H_
#define FINEFLOW_CORE_COMMON_ERROR_UTIL_H_
#include <string>

#include "fineflow/core/common/error.h"
#include "fineflow/core/common/result.hpp"
namespace fineflow {

std::string FormatErrorStr(const std::shared_ptr<StackedError>& error);
inline std::string FormatErrorStr(const Error& error) { return FormatErrorStr(error.stackedError()); }
}  // namespace fineflow
#endif
