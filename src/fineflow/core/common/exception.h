#ifndef FINEFLOW_CORE_COMMON_EXCEPTION_H_
#define FINEFLOW_CORE_COMMON_EXCEPTION_H_
#include <exception>
#include <string>
#include <utility>
namespace fineflow {
class Exception : public std::exception {
public:
  explicit Exception(std::string what) : what_(std::move(what)) {}
  ~Exception() override = default;

  [[nodiscard]] const char* what() const noexcept override { return what_.c_str(); }

private:
  std::string what_;
};

class RuntimeException : public Exception {
public:
  using Exception::Exception;
};

class TypeException : public Exception {
public:
  using Exception::Exception;
};

class IndexException : public Exception {
public:
  using Exception::Exception;
};

class NotImplementedException : public Exception {
public:
  using Exception::Exception;
};

}  // namespace fineflow
#endif  // FINEFLOW_CORE_COMMON_EXCEPTION_H_
