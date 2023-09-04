#ifndef FINEFLOW_CORE_COMMON_ERROR_H_
#define FINEFLOW_CORE_COMMON_ERROR_H_

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "fineflow/core/common/error.pb.h"
#include "fineflow/core/common/hash.hpp"
#include "fineflow/core/common/preprocess.h"
#include "fineflow/core/common/result_types.h"
namespace fineflow {
inline std::string RemoveProjectPathPrefix(const std::string& filename) {
#if defined(FINEFLOW_SOURCE_DIR) && defined(FINEFLOW_BINARY_DIR)
  std::string project_path = FF_PP_STRINGIZE(FINEFLOW_SOURCE_DIR);
  std::string project_build_path = FF_PP_STRINGIZE(FINEFLOW_BINARY_DIR);
  if (absl::StartsWith(filename, project_build_path)) {
    return std::filesystem::relative(filename, project_build_path);
  }
  if (absl::StartsWith(filename, project_path)) {
    return std::filesystem::relative(filename, project_path);
  }
#endif
  return filename;
}

class ErrorStackFrame final {
public:
  ErrorStackFrame(const ErrorStackFrame&) = default;
  ErrorStackFrame(const std::string& file, int64_t line, const std::string& function)
      : file_(RemoveProjectPathPrefix(file)), line_(line), function_(std::move(function)) {}
  ErrorStackFrame(const std::string& file, int64_t line, const std::string& function, const std::string& code_text)
      : file_(RemoveProjectPathPrefix(file)),
        line_(line),
        function_(std::move(function)),
        code_text_(std::move(code_text)) {}

  bool operator==(const ErrorStackFrame& other) const {
    return this->file_ == other.file_ && this->line_ == other.line_ && this->function_ == other.function_ &&
           this->code_text_ == other.code_text_;
  }

  [[nodiscard]] const std::string& file() const { return file_; }
  [[nodiscard]] int64_t line() const { return line_; }
  [[nodiscard]] const std::string& function() const { return function_; }
  [[nodiscard]] const std::string& codeText() const { return code_text_; }

  [[nodiscard]] std::string debugString() const {
    return file_ + ":" + std::to_string(line_) + " " + function_ + "\n\t" + code_text_ + "\n";
  }

private:
  std::string file_;
  int64_t line_;
  std::string function_;
  std::string code_text_;
};

class StackedError final {
public:
  StackedError();
  StackedError(const StackedError&) = default;

  // constexpr static int kStackReservedSize = 16;
  using FrameVector = std::vector<ErrorStackFrame>;

  const ErrorProto* operator->() const { return errorProto().get(); }
  ErrorProto* operator->() { return errorProtoMut(); }

  // Getters
  [[nodiscard]] const FrameVector& stackFrame() const { return stack_frame_; }
  [[nodiscard]] const std::shared_ptr<const ErrorProto>& errorProto() const { return error_proto_; }
  [[nodiscard]] std::string debugString() const {
    std::ostringstream ss;
    for (const auto& frame : stackFrame()) {
      ss << frame.debugString() << std::endl;
    }
    ss << errorProto()->DebugString();
    return ss.str();
  }

  // Setters
  void addStackFrame(const ErrorStackFrame& error_frame) { stack_frame_.push_back(error_frame); }
  class ErrorProto* errorProtoMut() { return const_cast<class ErrorProto*>(error_proto_.get()); }

private:
  FrameVector stack_frame_;
  std::shared_ptr<const class ErrorProto> error_proto_;
};

std::string GetErrorString(const std::shared_ptr<StackedError>& error);

class Error final {
public:
  Error(const std::shared_ptr<StackedError>& stacked_error)  // NOLINT
      : stacked_error_(stacked_error), msg_collecting_mode_(kMergeMessage) {}

  Error(const Error&) = default;
  ~Error() = default;

  [[nodiscard]] std::shared_ptr<StackedError> stackedError() const { return stacked_error_; }
  const ErrorProto* operator->() const { return stacked_error_->errorProto().get(); }
  ErrorProto* operator->() { return stacked_error_->errorProtoMut(); }
  template <class T>
  inline operator Maybe<T, Error>() {  // NOLINT
    return Failure<Error>(std::move(*this));
  }
  void assign(const Error& other) { stacked_error_ = other.stacked_error_; }
  void merge(const Error& other);

  Error&& addStackFrame(const ErrorStackFrame& error_stack_frame);
  // Error&& getStackTrace(int64_t depth = 32, int64_t skip_n_firsts = 2);

  // NOLINTBEGIN: readability-identifier-naming
  static Error Ok();
  // static Error ProtoParseFailedError();
  // static Error JobSetEmptyError();
  // static Error DeviceTagNotFoundError();
  static Error InvalidValueError();
  static Error IndexError();
  static Error TypeError();
  static Error TimeoutError();
  // static Error JobNameExistError();
  // static Error JobNameEmptyError();
  // static Error JobNameNotEqualError();
  // static Error NoJobBuildAndInferCtxError();
  // static Error JobConfFrozenError();
  // static Error JobConfNotSetError();
  // static Error JobConfRepeatedSetError();
  // static Error JobTypeNotSetError();
  // static Error LogicalBlobNameNotExistError();
  // static Error LogicalBlobNameExistError();
  // static Error LogicalBlobNameInvalidError();
  // static Error OpNameExistError();
  // static Error OpConfDeviceTagNoSetError();
  // static Error PlacementError();
  // static Error BlobSplitAxisInferError();
  // static Error UnknownJobBuildAndInferError();
  static Error CheckFailedError();
  static Error ValueNotFoundError();
  static Error TodoError();
  static Error UnimplementedError();
  static Error RuntimeError();
  // static Error OutOfMemoryError();
  // static Error BoxingNotSupportedError();
  // static Error MemoryZoneOutOfMemoryError(int64_t machine_id, int64_t mem_zone_id, uint64_t calc,
  //                                         uint64_t available, const std::string& device_type);
  // static Error OpKernelNotFoundError(const std::vector<std::string>& error_msgs);
  // static Error MultipleOpKernelsMatchedError(const std::vector<std::string>& error_msgs);
  // static Error LossBlobNotFoundError();

  // static Error RwMutexedObjectNotFoundError();

  // gradient
  // static Error GradientFunctionNotFoundError();

  // symbol
  // static Error SymbolIdUninitializedError();

  // static Error CompileOptionWrongError();

  // static Error InputDeviceNotMatchError();
  // NOLINTEND: readability-identifier-naming

  enum MsgCollectingMode {
    kInvalidMsgCollectingMode = 0,
    kMergeMessage,
    kOverrideThenMergeMessage,
  };

  [[nodiscard]] MsgCollectingMode msgCollectingMode() const { return msg_collecting_mode_; }
  void msgCollectingMode(enum MsgCollectingMode val) { msg_collecting_mode_ = val; }

private:
  std::shared_ptr<StackedError> stacked_error_;
  enum MsgCollectingMode msg_collecting_mode_ {};
};

void ThrowError(const std::shared_ptr<StackedError>& error);
inline void ThrowError(const Error& error) { ThrowError(error.stackedError()); }
const std::shared_ptr<StackedError>& ThreadLocalError();

inline Error& operator<<(Error& error, Error::MsgCollectingMode mode) {
  error.msgCollectingMode(mode);
  return error;
}

template <typename T>
Error& operator<<(Error& error, const T& x) {
  std::ostringstream ss;
  ss << x;
  if (error.msgCollectingMode() == Error::kMergeMessage) {
    error->set_msg(error->msg() + ss.str());
  } else if (error.msgCollectingMode() == Error::kOverrideThenMergeMessage) {
    error->set_msg(ss.str());
    error.msgCollectingMode(Error::kMergeMessage);
  } else {
    // GLOGLOGFATAL("UNIMPLEMENTED");
  }
  return error;
}

// r-value reference is used to supporting expressions like `Error() << "invalid value"`
template <typename T>
Error&& operator<<(Error&& error, const T& x) {
  error << x;
  return std::move(error);
}

template <>
inline Error&& operator<<(Error&& error, const std::stringstream& x) {
  error << x.str();
  return std::move(error);
}

template <>
inline Error&& operator<<(Error&& error, const std::ostream& x) {
  error << x.rdbuf();
  return std::move(error);
}

template <>
inline Error&& operator<<(Error&& error, const Error& x) {
  error.merge(x);
  return std::move(error);
}

// handle CHECK_OR_THROW(expr) << ... << std::endl;
inline Error&& operator<<(Error&& error, std::ostream& (*os)(std::ostream&)) {
  error << os;
  return std::move(error);
}

// extern const char* kOfBugIssueUploadPrompt;

}  // namespace fineflow
namespace std {

template <>
struct hash<::fineflow::ErrorStackFrame> final {
  size_t operator()(const ::fineflow::ErrorStackFrame& frame) const {
    return fineflow::Hash(frame.file(), frame.line(), frame.function(), frame.codeText());
  }
};
}  // namespace std
#endif
