#include "fineflow/core/common/error.h"

#include <iostream>

#include "fineflow/core/common/error_util.h"
namespace fineflow {

StackedError::StackedError() : error_proto_(new ErrorProto()) {}

namespace {

void LogError(const Error& error) {
  // gdb break point
  std::cout << error->msg() << std::endl;
}

std::shared_ptr<StackedError>* MutThreadLocalError() {
  thread_local std::shared_ptr<StackedError> error;
  return &error;
}

}  // namespace

Error&& Error::addStackFrame(const ErrorStackFrame& error_stack_frame) {
  stacked_error_->addStackFrame(error_stack_frame);
  return std::move(*this);
}

// Error&& Error::getStackTrace(int64_t /*depth*/, int64_t /*skip_n_firsts*/) {
// backward::StackTrace st;
// backward::SnippetFactory snippets;
// backward::TraceResolver resolver;
// st.load_here(depth);
// st.skip_n_firsts(skip_n_firsts);
// resolver.load_stacktrace(st);

// for (int i = 0; i < st.size(); i++) {
// const auto& trace = resolver.resolve(st[i]);
// if (!backward::Printer::is_oneflow_file(trace.object_filename)) {
//   continue;
// }

//  without debug info
// if (!trace.source.filename.size()) {
//   stacked_error_->addStackFrame(ErrorStackFrame(trace.object_filename, -1, trace.object_function));
// }

//  with debug info
// if (trace.source.filename.size()) {
//   const backward::ResolvedTrace::SourceLoc& source_loc = trace.source;
//   backward::SnippetFactory::lines_t lines =
//       snippets.get_snippet(source_loc.filename, source_loc.line, static_cast<unsigned>(1));
//   std::string code_text = lines[0].second;
//   const auto pos = code_text.find_first_not_of(" \t");
//   code_text = code_text.substr(pos, code_text.size() - pos);
//   stacked_error_->add_stack_frame(
//       SymbolOf(ErrorStackFrame(source_loc.filename, source_loc.line, source_loc.function, code_text)));
// }

// for (size_t inliner_idx = 0; inliner_idx < trace.inliners.size(); ++inliner_idx) {
// const backward::ResolvedTrace::SourceLoc& source_loc = trace.inliners[inliner_idx];
// backward::SnippetFactory::lines_t lines =
//     snippets.get_snippet(source_loc.filename, source_loc.line, static_cast<unsigned>(1));
// std::string code_text = lines[0].second;
// const auto pos = code_text.find_first_not_of(" \t");
// code_text = code_text.substr(pos, code_text.size() - pos);
//   stacked_error_->addStackFrame(
//       SymbolOf(ErrorStackFrame(source_loc.filename, source_loc.line, source_loc.function, code_text)));
// }
// }
//   return std::move(*this);
// }

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

std::string GetStackedErrorString(const std::shared_ptr<StackedError>& error) {
  const auto& error_str = FormatErrorStr(error).transform_error([&](auto) { return error->debugString(); }).value();
  // const auto& error_str = maybe_error.GetDataAndStackedError(error->DebugString());
  // CHECK_NE(error->errorProto()->error_type_case(), ErrorProto::ERROR_TYPE_NOT_SET);
  return error_str;
}

std::string GetErrorString(const std::shared_ptr<StackedError>& error) {
  std::string error_str;
  if (true) {  // NOLINT
    error_str = GetStackedErrorString(error);
  } else {
    error_str = error->errorProto()->msg();
  }
  if (error_str.empty()) {
    error_str = "<No error message>";
  }
  return error_str;
}

// void ThrowError(const std::shared_ptr<StackedError>& error) {
//   std::string error_str;
//   fmt::format_to(std::back_inserter(error_str), "{}: {}\n",
//                  fmt::styled("Error", fmt::emphasis::bold | fmt::fg(fmt::color::red)), GetErrorString(error));
//   // Append foreign stack trace (e.g. Python stack trace) when it is available.
//   if (ForeignFrameThreadLocalGuard::Current().has_value()) {
//     auto frame = *CHECK_JUST(ForeignFrameThreadLocalGuard::Current());
//     if (!IsMainThread()) {
//       if (auto* stack_getter = Singleton<ForeignStackGetter>::Get()) {
//         fmt::format_to(std::back_inserter(error_str), fmt::emphasis::bold | fmt::fg(fmt::color::dark_orange),
//                        "Related Python stack trace:");
//         if (IsPythonStackGetterEnabledByDebugBuild()) {
//           fmt::format_to(std::back_inserter(error_str),
//                          " (You are seeing this stack trace because you compiled OneFlow with "
//                          "CMAKE_BUILD_TYPE=Debug. If you want to see it even with other CMAKE_BUILD_TYPEs, "
//                          "you can set ONEFLOW_DEBUG or ONEFLOW_PYTHON_STACK_GETTER to 1)");
//         }
//         fmt::format_to(std::back_inserter(error_str), "\n{}", stack_getter->GetFormattedStack(frame));
//       } else {
//         fmt::format_to(
//             std::back_inserter(error_str), "You can set {} or {} to 1 to get the Python stack of the error.",
//             fmt::styled("ONEFLOW_DEBUG", fmt::emphasis::bold | fmt::fg(fmt::color::dark_orange)),
//             fmt::styled("ONEFLOW_PYTHON_STACK_GETTER", fmt::emphasis::bold | fmt::fg(fmt::color::dark_orange)));
//       }
//     }
//   }
//   *MutThreadLocalError() = error;
//   if ((*error)->has_runtime_error()) {
//     throw RuntimeException(error_str);
//   }
//   if ((*error)->has_type_error()) {
//     throw TypeException(error_str);
//   }
//   if ((*error)->has_index_error()) {
//     throw IndexException(error_str);
//   }
//   if ((*error)->has_unimplemented_error()) {
//     throw NotImplementedException(error_str);
//   }
//   throw Exception(GetStackedErrorString(error));
// }

const std::shared_ptr<StackedError>& ThreadLocalError() { return *MutThreadLocalError(); }

// const char* kOfBugIssueUploadPrompt =
//     "This is a oneflow bug, please submit an issue at "
//     "'https://github.com/Oneflow-Inc/oneflow/issues' including "
//     "the log information of the error, the "
//     "minimum reproduction code, and the system information.";

}  // namespace fineflow
