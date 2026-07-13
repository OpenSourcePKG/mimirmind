#include "core/log/Log.hpp"

#include "core/config/Config.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

namespace mimirmind::runtime {

namespace {

std::atomic<LogLevel> g_level{LogLevel::Info};
std::mutex            g_mutex;
std::ofstream         g_file;

constexpr std::string_view levelTag(LogLevel lvl) noexcept {
    switch (lvl) {
        case LogLevel::Trace: return "TRACE";
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO ";
        case LogLevel::Warn:  return "WARN ";
        case LogLevel::Error: return "ERROR";
        case LogLevel::Off:   return "OFF  ";
    }
    return "?????";
}

bool eqIgnoreCase(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        char ca = a[i];
        char cb = b[i];
        if (ca >= 'A' && ca <= 'Z') ca = static_cast<char>(ca + ('a' - 'A'));
        if (cb >= 'A' && cb <= 'Z') cb = static_cast<char>(cb + ('a' - 'A'));
        if (ca != cb) {
            return false;
        }
    }
    return true;
}

LogLevel parseLevel(std::string_view s) noexcept {
    if (eqIgnoreCase(s, "trace"))   return LogLevel::Trace;
    if (eqIgnoreCase(s, "debug"))   return LogLevel::Debug;
    if (eqIgnoreCase(s, "info"))    return LogLevel::Info;
    if (eqIgnoreCase(s, "warn"))    return LogLevel::Warn;
    if (eqIgnoreCase(s, "warning")) return LogLevel::Warn;
    if (eqIgnoreCase(s, "error"))   return LogLevel::Error;
    if (eqIgnoreCase(s, "err"))     return LogLevel::Error;
    if (eqIgnoreCase(s, "off"))     return LogLevel::Off;
    if (eqIgnoreCase(s, "none"))    return LogLevel::Off;
    if (eqIgnoreCase(s, "silent"))  return LogLevel::Off;
    return LogLevel::Info;
}

// Keep paths readable in log lines: strip everything before "src/" if
// present, otherwise reduce to the basename.
std::string_view shortFile(std::string_view path) noexcept {
    const auto p = path.rfind("/src/");
    if (p != std::string_view::npos) {
        return path.substr(p + 1);
    }
    const auto s = path.rfind('/');
    if (s != std::string_view::npos) {
        return path.substr(s + 1);
    }
    return path;
}

std::string formatTimestamp() {
    using namespace std::chrono;
    const auto now           = system_clock::now();
    const auto since_epoch   = now.time_since_epoch();
    const auto ms_total      = duration_cast<milliseconds>(since_epoch).count();
    const auto secs          = ms_total / 1000;
    const auto ms_part       = ms_total % 1000;
    const std::time_t tt     = static_cast<std::time_t>(secs);

    std::tm tm{};
    gmtime_r(&tt, &tm);

    // 78-byte worst case from snprintf (large year + 3-digit ms); 96 leaves
    // comfortable headroom without affecting performance.
    char buf[96];
    std::snprintf(buf, sizeof(buf),
        "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ",
        tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
        tm.tm_hour, tm.tm_min, tm.tm_sec,
        static_cast<long>(ms_part));
    return std::string{buf};
}

} // namespace

void Log::setLevel(LogLevel lvl) noexcept {
    g_level.store(lvl, std::memory_order_relaxed);
}

LogLevel Log::level() noexcept {
    return g_level.load(std::memory_order_relaxed);
}

bool Log::setFile(std::string_view path) {
    std::scoped_lock lock{g_mutex};
    if (g_file.is_open()) {
        g_file.close();
    }
    if (path.empty()) {
        return true;
    }
    g_file.open(std::string{path}, std::ios::out | std::ios::app);
    return g_file.is_open();
}

void Log::initFromConfig(const LogSettings& settings) {
    if (!settings.level.empty()) {
        setLevel(parseLevel(settings.level));
    }
    if (!settings.file.empty()) {
        setFile(settings.file);
    }
}

void Log::write(LogLevel lvl,
                std::string_view tag,
                std::string_view msg,
                const std::source_location& loc) noexcept {
    if (!enabled(lvl)) {
        return;
    }
    try {
        const std::string line = std::format(
            "{} [{}] [{:<16}] {} ({}:{})\n",
            formatTimestamp(),
            levelTag(lvl),
            tag,
            msg,
            shortFile(loc.file_name()),
            loc.line());

        std::scoped_lock lock{g_mutex};
        std::clog << line;
        std::clog.flush();
        if (g_file.is_open()) {
            g_file << line;
            g_file.flush();
        }
    } catch (...) {
        // logging must never throw
    }
}

} // namespace mimirmind::runtime