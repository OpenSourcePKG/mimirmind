// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <format>
#include <source_location>
#include <string_view>
#include <utility>

namespace mimirmind::core::config {
struct LogSettings;
}

namespace mimirmind::core::log {

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
 * to a file. Configured from `Config.server.log` at process start via
 * `initFromConfig()`; `server.log.level` accepts trace|debug|info|warn|
 * error|off (default: info), `server.log.file` is a path (empty = no file
 * sink).
 *
 * Use the MM_LOG_* macros below — they capture std::source_location and
 * skip formatting when the level is filtered out.
 */
class Log {
public:
    static void initFromConfig(const ::mimirmind::core::config::LogSettings& settings);

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
} // namespace mimirmind::core::log

#define MM_LOG_TRACE(tag, ...) \
    ::mimirmind::core::log::detail::logFormat( \
        ::mimirmind::core::log::LogLevel::Trace, (tag), \
        std::source_location::current(), __VA_ARGS__)

#define MM_LOG_DEBUG(tag, ...) \
    ::mimirmind::core::log::detail::logFormat( \
        ::mimirmind::core::log::LogLevel::Debug, (tag), \
        std::source_location::current(), __VA_ARGS__)

#define MM_LOG_INFO(tag, ...) \
    ::mimirmind::core::log::detail::logFormat( \
        ::mimirmind::core::log::LogLevel::Info, (tag), \
        std::source_location::current(), __VA_ARGS__)

#define MM_LOG_WARN(tag, ...) \
    ::mimirmind::core::log::detail::logFormat( \
        ::mimirmind::core::log::LogLevel::Warn, (tag), \
        std::source_location::current(), __VA_ARGS__)

#define MM_LOG_ERROR(tag, ...) \
    ::mimirmind::core::log::detail::logFormat( \
        ::mimirmind::core::log::LogLevel::Error, (tag), \
        std::source_location::current(), __VA_ARGS__)
/* trailing newline intentional — silences -Wbackslash-newline-escape at EOF */