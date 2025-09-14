#ifndef FINEFLOW_CORE_COMMON_LOG_H_
#define FINEFLOW_CORE_COMMON_LOG_H_

#define SPDLOG_LOGGER_STREAM(log, lvl)                                                                \
  log && log->should_log(lvl) &&                                                                      \
      fineflow::LogStream(default_logger(), lvl,                                                      \
                          source_loc(__FILE__, __LINE__, static_cast<const char *>(__FUNCTION__))) <= \
          std::ostringstream()

#define LOG(x) SPDLOG_LOGGER_STREAM(default_logger(), x)
#endif
