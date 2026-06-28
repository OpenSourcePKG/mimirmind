#pragma once

#include <format>
#include <source_location>
#include <string_view>
#include <utility>

namespace mimirmind::runtime {

enum class LogLevel : int {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
    Off   = 5,
};

/**
 * Process-wide logger. Sinks to stderr unconditionally, optionally also
 * to a file. Level- and sink-controllable through env at process start:
 *
 *   MIMIRMIND_LOG_LEVEL = trace|debug|info|warn|error|off    (default: info)
 *   MIMIRMIND_LOG_FILE  = /path/to/log                       (default: none)
 *
 * Use the MM_LOG_* macros below — they capture std::source_location and
 * skip formatting when the level is filtered out.
 */
class Log {
public:
    static void initFromEnv();

    static void setLevel(LogLevel lvl) noexcept;
    [[nodiscard]] static LogLevel level() noexcept;

    static bool setFile(std::string_view path);

    static void write(LogLevel lvl,
                      std::string_view tag,
                      std::string_view msg,
                      const std::source_location& loc) noexcept;

    [[nodiscard]] static bool enabled(LogLevel lvl) noexcept {
        return static_cast<int>(lvl) >= static_cast<int>(level());
    }
};

namespace detail {

template <typename... Args>
inline void logFormat(LogLevel lvl,
                      std::string_view tag,
                      const std::source_location& loc,
                      std::format_string<Args...> fmt,
                      Args&&... args) noexcept {
    if (!Log::enabled(lvl)) {
        return;
    }
    try {
        Log::write(lvl, tag,
                   std::format(fmt, std::forward<Args>(args)...),
                   loc);
    } catch (...) {
        Log::write(lvl, tag,
                   std::string_view{"<log formatting failed>"},
                   loc);
    }
}

} // namespace detail
} // namespace mimirmind::runtime

#define MM_LOG_TRACE(tag, ...) \
    ::mimirmind::runtime::detail::logFormat( \
        ::mimirmind::runtime::LogLevel::Trace, (tag), \
        std::source_location::current(), __VA_ARGS__)

#define MM_LOG_DEBUG(tag, ...) \
    ::mimirmind::runtime::detail::logFormat( \
        ::mimirmind::runtime::LogLevel::Debug, (tag), \
        std::source_location::current(), __VA_ARGS__)

#define MM_LOG_INFO(tag, ...) \
    ::mimirmind::runtime::detail::logFormat( \
        ::mimirmind::runtime::LogLevel::Info, (tag), \
        std::source_location::current(), __VA_ARGS__)

#define MM_LOG_WARN(tag, ...) \
    ::mimirmind::runtime::detail::logFormat( \
        ::mimirmind::runtime::LogLevel::Warn, (tag), \
        std::source_location::current(), __VA_ARGS__)

#define MM_LOG_ERROR(tag, ...) \
    ::mimirmind::runtime::detail::logFormat( \
        ::mimirmind::runtime::LogLevel::Error, (tag), \
        std::source_location::current(), __VA_ARGS__)
/* trailing newline intentional — silences -Wbackslash-newline-escape at EOF */