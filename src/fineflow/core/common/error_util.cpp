#include "fineflow/core/common/error_util.h"
namespace fineflow {
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
    error_msg = *r.transform_error([&](auto) { return error_msg; });
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
  const google::protobuf::Descriptor* error_des = error_proto->GetDescriptor();
  const google::protobuf::OneofDescriptor* oneof_field_des = error_des->FindOneofByName("error_type");
  const google::protobuf::Reflection* error_ref = error_proto->GetReflection();
  const google::protobuf::FieldDescriptor* field_des =
      error_ref->GetOneofFieldDescriptor(*error_proto, oneof_field_des);
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
  auto r = *details::FormatMsgOfErrorType(error).transform_error([](auto) { return "unknown error."; });
  if (!r.empty()) {
    ss << std::endl << r;
  }
  return ss.str();
}

}  // namespace fineflow
