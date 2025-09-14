module;
// #include "fineflow/core/common/preprocess.h"
#include "fineflow/core/common/result.h"
export module fineflow.core.common.error;
import std;
import std.compat;
import fineflow.core.common.exception;
import fineflow.core.common.absl_match;
import fineflow.core.common.error_proto;
import fineflow.core.common.hash;
import fineflow.core.common.fmt;
import fineflow.core.common.data_type_proto;

export namespace fineflow {

template <typename T1, typename T2>
using expected = std::expected<T1, T2>;

template <typename E>
using unexpected = std::unexpected<E>;

template <class T, class E>
using Maybe = expected<T, E>;

template <class E>
using Failure = unexpected<E>;

class Error;
template <class T, class E = Error>
using Ret = Maybe<T, E>;

template <class E = Error>
inline Failure<E> Fail(E&& e) {
  return Failure<E>(std::forward<E>(e));
}

struct Ok {
  template <class T>
  inline operator Ret<T>() {  // NOLINT
    return {};
  }
};

inline std::string RemoveProjectPathPrefix(const std::string& filename) {
#if defined(FINEFLOW_SOURCE_DIR) && defined(FINEFLOW_BINARY_DIR)
  std::string project_path = FF_PP_STRINGIZE(FINEFLOW_SOURCE_DIR);
  std::string project_build_path = FF_PP_STRINGIZE(FINEFLOW_BINARY_DIR);
  if (StartsWith(filename, project_build_path)) {
    return std::filesystem::relative(filename, project_build_path);
  }
  if (StartsWith(filename, project_path)) {
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
  [[nodiscard]] const std::shared_ptr<ErrorProto>& errorProto() const { return error_proto_; }
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
  ErrorProto* errorProtoMut() { return const_cast<ErrorProto*>(error_proto_.get()); }

private:
  FrameVector stack_frame_;
  std::shared_ptr<ErrorProto> error_proto_;
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

StackedError::StackedError() : error_proto_(new ErrorProto()) {}

// namespace {

void LogError(const Error& error) {
  // gdb break point
  std::cout << error->msg() << std::endl;
}

std::shared_ptr<StackedError>* MutThreadLocalError() {
  thread_local std::shared_ptr<StackedError> error;
  return &error;
}

// }  // namespace

Error&& Error::addStackFrame(const ErrorStackFrame& error_stack_frame) {
  stacked_error_->addStackFrame(error_stack_frame);
  return std::move(*this);
}

void Error::merge(const Error& other) {
  auto* error_proto = stacked_error_->errorProtoMut();
  error_proto->MergeFrom(*other.stacked_error_->errorProto());
}

Error Error::Ok() { return std::make_shared<StackedError>(); }

Error Error::InvalidValueError() {
  auto error = std::make_shared<StackedError>();
  error->errorProtoMut()->mutable_invalid_value_error();
  return error;
}

Error Error::IndexError() {
  auto error = std::make_shared<StackedError>();
  error->errorProtoMut()->mutable_index_error();
  return error;
}

Error Error::TypeError() {
  auto error = std::make_shared<StackedError>();
  error->errorProtoMut()->mutable_type_error();
  return error;
}

Error Error::TimeoutError() {
  auto error = std::make_shared<StackedError>();
  error->errorProtoMut()->mutable_timeout_error();
  return error;
}

Error Error::ValueNotFoundError() {
  auto error = std::make_shared<StackedError>();
  error->errorProtoMut()->mutable_value_not_found_error();
  return error;
}

Error Error::TodoError() {
  auto error = std::make_shared<StackedError>();
  error->errorProtoMut()->mutable_todo_error();
  return error;
}

Error Error::UnimplementedError() {
  auto error = std::make_shared<StackedError>();
  error->errorProtoMut()->mutable_unimplemented_error();
  return error;
}

Error Error::RuntimeError() {
  auto error = std::make_shared<StackedError>();
  error->errorProtoMut()->mutable_runtime_error();
  return error;
}

Error Error::CheckFailedError() {
  auto error = std::make_shared<StackedError>();
  error->errorProtoMut()->mutable_check_failed_error();
  return error;
}

std::string FormatErrorStr(const std::shared_ptr<StackedError>& error);
inline std::string FormatErrorStr(const Error& error) { return FormatErrorStr(error.stackedError()); }

std::string GetErrorString(const std::shared_ptr<StackedError>& error) {
  std::string error_str;
#ifdef DEBUG
  error_str = FormatErrorStr(error);
#else
  error_str = error->errorProto()->msg();
#endif
  if (error_str.empty()) {
    error_str = "<No error message>";
  }
  return error_str;
}

void ThrowError(const std::shared_ptr<StackedError>& error) {
  std::string error_str;
  // fmt::format_to(std::back_inserter(error_str), "{}: {}",
  //                fmt::styled("Error", fmt::emphasis::bold | fmt::fg(fmt::color::red)), GetErrorString(error));
  format_to(std::back_inserter(error_str), "{}: {}", "Error", GetErrorString(error));
  *MutThreadLocalError() = error;

  if ((*error)->has_type_error()) {
    throw TypeException(error_str);
  }
  if ((*error)->has_index_error()) {
    throw IndexException(error_str);
  }
  if ((*error)->has_unimplemented_error()) {
    throw NotImplementedException(error_str);
  }
  throw RuntimeException(error_str);
}

const std::shared_ptr<StackedError>& ThreadLocalError() { return *MutThreadLocalError(); }

namespace details {

std::string StripSpace(std::string str) {
  if (str.empty()) {
    return "";
  }
  size_t pos = str.find_first_not_of(' ');
  if (pos != std::string::npos) {
    str.erase(0, pos);
  }
  pos = str.find_last_not_of(' ');
  if (pos != std::string::npos) {
    str.erase(pos + 1);
  }
  return str;
}

bool IsLetterNumberOrUnderline(char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_');
}

Ret<std::string> ShortenMsg(std::string str) {
  // 150 characters is the threshold
  const int num_character_threshold = 150;
  const int num_displayed_character = 50;
  if (str.empty()) {
    return str;
  }
  str = StripSpace(str);
  if (str.size() < num_character_threshold) {
    return str;
  }

  // left part whose number of characters is just over 50
  int left_index = num_displayed_character;
  bool pre_condition = IsLetterNumberOrUnderline(str.at(left_index));
  for (; left_index < str.size(); left_index++) {
    bool cur_condition = IsLetterNumberOrUnderline(str.at(left_index));
    if ((pre_condition && !cur_condition) || (!pre_condition && cur_condition)) {
      break;
    }
  }

  // right part whose number of characters is just over 50
  int right_index = str.size() - num_displayed_character;
  pre_condition = IsLetterNumberOrUnderline(str.at(right_index));
  for (; right_index >= 0; right_index--) {
    bool cur_condition = IsLetterNumberOrUnderline(str.at(right_index));
    if ((pre_condition && !cur_condition) || (!pre_condition && cur_condition)) {
      right_index++;
      break;
    }
  }
  // a long word of more than 150
  if (right_index - left_index < 50) {
    return str;
  }
  std::stringstream ss;
  CHECK_OR_RETURN(left_index >= 0);
  CHECK_OR_RETURN(left_index < str.size());
  ss << str.substr(0, left_index);
  ss << " ... ";
  CHECK_OR_RETURN(right_index >= 0);
  CHECK_OR_RETURN(right_index < str.size());
  ss << str.substr(right_index);
  return ss.str();
}

// file info in stack frame
std::string FormatFileOfStackFrame(const std::string& file) {
  std::stringstream ss;
  ss << "\n  File \"" << file << "\", ";
  return ss.str();
}

// line info in stack frame
std::string FormatLineOfStackFrame(const int64_t& line) {
  std::stringstream ss;
  if (line >= 0) {
    ss << "line " << line << ",";
  } else {
    ss << "line <unknown>,";
  }
  return ss.str();
}

// function info in stack frame
std::string FormatFunctionOfStackFrame(const std::string& function) {
  std::stringstream ss;
  ss << " in " << function;
  return ss.str();
}

// msg in stack frame
std::string FormatMsgOfStackFrame(std::string error_msg, bool is_last_stackFrame) {
  const bool debug_mode = true;
  // only shorten the message if it is not the last stack frame AND not in debug mode
  if (!is_last_stackFrame && !debug_mode) {
    Ret<std::string> r = ShortenMsg(error_msg);
    if (r) {
      error_msg = r.value();
    }
  }
  // error_msg of last stack frame come from "<<"
  if (is_last_stackFrame) {
    error_msg = StripSpace(error_msg);
  }
  std::stringstream ss;
  if (!error_msg.empty()) {
    ss << "\n    " << error_msg;
  }
  return ss.str();
}

// the msg in error type instance.
Ret<std::string> FormatMsgOfErrorType(const std::shared_ptr<StackedError>& error) {
  const auto& error_proto = error->errorProto();
  CHECK_NE_OR_RETURN(error_proto->error_type_case(), ErrorProto::ERROR_TYPE_NOT_SET)
      << Error::RuntimeError() << "Parse error failed, unknown error type";
  std::stringstream ss;
  // const google::protobuf::Descriptor* error_des = error_proto->GetDescriptor();
  // const google::protobuf::OneofDescriptor* oneof_field_des = error_des->FindOneofByName("error_type");
  // const google::protobuf::Reflection* error_ref = error_proto->GetReflection();
  // const google::protobuf::FieldDescriptor* field_des =
  //     error_ref->GetOneofFieldDescriptor(*error_proto, oneof_field_des);
  const Descriptor* error_des = error_proto->GetDescriptor();
  const OneofDescriptor* oneof_field_des = error_des->FindOneofByName("error_type");
  const Reflection* error_ref = error_proto->GetReflection();
  const FieldDescriptor* field_des = error_ref->GetOneofFieldDescriptor(*error_proto, oneof_field_des);
  CHECK_OR_RETURN(field_des != nullptr);
  ss << "Error Type: " << field_des->full_name();
  return ss.str();
}

}  // namespace details

std::string FormatErrorStr(const std::shared_ptr<StackedError>& error) {
  std::stringstream ss;
  ss << error->errorProto()->msg();
  ss << error->errorProto()->frame_msg();
  // Get msg from stack frame of error proto
  for (auto iter = error->stackFrame().rbegin(); iter < error->stackFrame().rend(); iter++) {
    const auto& stack_frame = *iter;
    ss << details::FormatFileOfStackFrame(stack_frame.file()) << details::FormatLineOfStackFrame(stack_frame.line())
       << details::FormatFunctionOfStackFrame(stack_frame.function());

    ss << details::FormatMsgOfStackFrame(stack_frame.codeText(), iter == error->stackFrame().rend() - 1);
  }
  // Get msg from error type of error proto
  auto r = details::FormatMsgOfErrorType(error).value_or("unknown error.");
  if (!r.empty()) {
    ss << std::endl << r;
  }
  return ss.str();
}

}  // namespace fineflow
namespace std {

template <>
struct hash<::fineflow::ErrorStackFrame> final {
  size_t operator()(const ::fineflow::ErrorStackFrame& frame) const {
    return fineflow::Hash(frame.file(), frame.line(), frame.function(), frame.codeText());
  }
};
}  // namespace std
