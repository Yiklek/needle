#ifndef FINEFLOW_CORE_COMMON_LOG_H_
#define FINEFLOW_CORE_COMMON_LOG_H_
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <sstream>

#include "spdlog/cfg/env.h"  // support for loading levels from the environment variable
#include "spdlog/spdlog.h"
namespace fineflow {
inline void Init() { spdlog::cfg::load_env_levels(); }
struct LogInit {
  LogInit() { Init(); }
};
static LogInit log_init;

#define SPDLOG_LOGGER_STREAM(log, lvl) \
  log && log->should_log(lvl) &&       \
      fineflow::LogStream(log, lvl, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}) <= std::ostringstream()

class LogStream {
  std::shared_ptr<spdlog::logger> logger_;
  spdlog::level::level_enum level_;
  spdlog::source_loc loc_;

public:
  LogStream(std::shared_ptr<spdlog::logger> log, spdlog::level::level_enum lvl, spdlog::source_loc loc)
      : logger_{log}, level_{lvl}, loc_{loc} {}
  bool operator<=(const std::ostringstream& line) {
    logger_->log(loc_, level_, "{}", line.str());
    return true;
  }
};

// specific log implementation macros. not use.
// you should use LOG in namespace fineflow
// if not, must specify namespace fineflow, such as fineflow::info

// #define TRACE spdlog::level::trace
// #define DEBUG spdlog::level::debug
// #define INFO spdlog::level::info
// #define WARN spdlog::level::warn
// #define ERROR spdlog::level::err
// #define CRITICAL spdlog::level::CRITICAL
// #define ERROR spdlog::level::err
// #define OFF spdlog::level::off
using namespace spdlog::level;

#define LOG(x) SPDLOG_LOGGER_STREAM(spdlog::default_logger(), x)
}  // namespace fineflow
#endif  // FINEFLOW_CORE_COMMON_LOG_H_
