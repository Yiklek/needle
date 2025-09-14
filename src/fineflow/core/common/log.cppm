module;
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <sstream>
#include <utility>

#include "spdlog/cfg/env.h"  // support for loading levels from the environment variable
#include "spdlog/spdlog.h"
export module fineflow.core.common.log;
import std;
export namespace fineflow {
inline void Init() { spdlog::cfg::load_env_levels(); }
struct LogInit {
  LogInit() { Init(); }
};
LogInit log_init;

auto default_logger = spdlog::default_logger;
using source_loc = spdlog::source_loc;

class LogStream {
  std::shared_ptr<spdlog::logger> logger_;
  spdlog::level::level_enum level_;
  spdlog::source_loc loc_;

public:
  LogStream(std::shared_ptr<spdlog::logger> log, spdlog::level::level_enum lvl, spdlog::source_loc loc)
      : logger_{std::move(log)}, level_{lvl}, loc_{loc} {}
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

#define SPDLOG_LOGGER_STREAM(log, lvl) \
  log && log->should_log(lvl) &&       \
      fineflow::LogStream(log, lvl, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}) <= std::ostringstream()

#define LOG(x) SPDLOG_LOGGER_STREAM(spdlog::default_logger(), x)
}  // namespace fineflow
