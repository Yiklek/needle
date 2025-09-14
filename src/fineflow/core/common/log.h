#ifndef FINEFLOW_CORE_COMMON_LOG_H_
#define FINEFLOW_CORE_COMMON_LOG_H_

#define SPDLOG_LOGGER_STREAM(logger, lvl)                                                            \
  logger && logger->should_log(fineflow::log::level::lvl) &&                                         \
      fineflow::LogStream(default_logger(), fineflow::log::level::lvl,                               \
                          source_loc(__FILE__, __LINE__, static_cast<const char*>(__FUNCTION__))) <= \
          std::ostringstream()

#define LOG(x) SPDLOG_LOGGER_STREAM(default_logger(), x)
#endif
