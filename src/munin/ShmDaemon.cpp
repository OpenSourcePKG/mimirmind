// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "munin/ShmDaemon.hpp"

#include "core/config/Config.hpp"
#include "core/log/Log.hpp"
#include "munin/ShmModelStore.hpp"
#include "munin/ShmSocketServer.hpp"

#include <sys/eventfd.h>
#include <sys/signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

namespace mimirmind::munin {

namespace {

std::atomic<int> g_shutdownEventFd{-1};
std::atomic<int> g_lastSignal{0};

void signalHandler(int sig) noexcept {
    g_lastSignal.store(sig);
    const int fd = g_shutdownEventFd.load();
    if (fd < 0) return;
    const std::uint64_t one = 1;
    ::ssize_t r = ::write(fd, &one, sizeof(one));
    (void)r;
}

bool installSignalHandlers() noexcept {
    struct ::sigaction sa{};
    sa.sa_handler = &signalHandler;
    sa.sa_flags   = 0;
    ::sigemptyset(&sa.sa_mask);
    if (::sigaction(SIGINT,  &sa, nullptr) < 0) return false;
    if (::sigaction(SIGTERM, &sa, nullptr) < 0) return false;

    struct ::sigaction sp{};
    sp.sa_handler = SIG_IGN;
    ::sigemptyset(&sp.sa_mask);
    if (::sigaction(SIGPIPE, &sp, nullptr) < 0) return false;
    return true;
}

// `mkdir -p` semantics for the socket's parent dir. Idempotent.
bool ensureParentDir(const std::string& path) noexcept {
    const auto slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return true;
    const std::string dir = path.substr(0, slash);
    std::string acc;
    acc.reserve(dir.size());
    for (std::size_t i = 0; i < dir.size(); ++i) {
        acc.push_back(dir[i]);
        if (dir[i] != '/' && i + 1 != dir.size()) continue;
        if (acc == "/") continue;
        if (::mkdir(acc.c_str(), 0755) < 0 && errno != EEXIST) return false;
    }
    return true;
}

} // namespace

int ShmDaemon::run(const CliOptions& opts) noexcept {
    // ---- config -------------------------------------------------------
    ::mimirmind::core::config::Config cfg;
    try {
        if (opts.configPath.empty()) {
            std::fprintf(stderr,
                         "munin-shm: --config is required (path to config.json)\n");
            return 2;
        }
        cfg = ::mimirmind::core::config::loadConfig(opts.configPath);
    } catch (const std::exception& x) {
        std::fprintf(stderr, "munin-shm: load config failed: %s\n", x.what());
        return 2;
    }

    if (!opts.logFile.empty())  cfg.server.log.file  = opts.logFile;
    if (!opts.logLevel.empty()) cfg.server.log.level = opts.logLevel;
    ::mimirmind::core::log::Log::initFromConfig(cfg.server.log);

    MM_LOG_INFO("munin",
                "munin-shm starting: config='{}' pid={}",
                opts.configPath, static_cast<int>(::getpid()));

    // ---- model store (memfd host RAM; no L0, no governor) -------------
    std::unique_ptr<ShmModelStore> store;
    try {
        store = std::make_unique<ShmModelStore>(cfg);
    } catch (const std::exception& x) {
        MM_LOG_ERROR("munin", "ShmModelStore init failed: {}", x.what());
        return 4;
    }

    // ---- socket path -------------------------------------------------
    std::string socketPath = opts.socketPath;
    if (socketPath.empty()) {
        socketPath = "/var/run/munin/munin.sock";
    }
    if (!ensureParentDir(socketPath)) {
        MM_LOG_ERROR("munin",
                     "cannot create parent directory for socket '{}': {}",
                     socketPath, std::strerror(errno));
        return 5;
    }

    // ---- signal wiring -----------------------------------------------
    const int evfd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (evfd < 0) {
        MM_LOG_ERROR("munin",
                     "eventfd creation failed: {} (errno={})",
                     std::strerror(errno), errno);
        return 6;
    }
    g_shutdownEventFd.store(evfd);
    if (!installSignalHandlers()) {
        MM_LOG_ERROR("munin",
                     "installing signal handlers failed: {} (errno={})",
                     std::strerror(errno), errno);
        ::close(evfd);
        return 6;
    }

    // ---- serve -------------------------------------------------------
    int exitCode = 0;
    try {
        ShmSocketServer server{*store, socketPath};
        server.serve(evfd);
    } catch (const std::exception& x) {
        MM_LOG_ERROR("munin", "ShmSocketServer failed: {}", x.what());
        exitCode = 7;
    }

    const int sig = g_lastSignal.load();
    if (sig != 0) {
        MM_LOG_INFO("munin",
                    "munin-shm shutdown initiated by signal {} ({})",
                    sig,
                    sig == SIGINT  ? "SIGINT"  :
                    sig == SIGTERM ? "SIGTERM" : "?");
    }

    g_shutdownEventFd.store(-1);
    ::close(evfd);

    MM_LOG_INFO("munin", "munin-shm exited (code={})", exitCode);
    return exitCode;
}

} // namespace mimirmind::munin
