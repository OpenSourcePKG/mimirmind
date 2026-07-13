#include "munin/Daemon.hpp"

#include "core/config/Config.hpp"
#include "core/l0/L0Context.hpp"
#include "core/l0/UsmAllocator.hpp"
#include "core/log/Log.hpp"
#include "munin/ModelStore.hpp"
#include "munin/SocketServer.hpp"

#include <sys/eventfd.h>
#include <sys/signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <thread>

namespace mimirmind::munin {

namespace {

// One process-global eventfd. The signal handler must be async-safe;
// eventfd's write(2) is on the async-safe list, so we route SIGINT/
// SIGTERM through it and let the main thread do the rest.
std::atomic<int> g_shutdownEventFd{-1};
std::atomic<int> g_lastSignal{0};

void signalHandler(int sig) noexcept {
    g_lastSignal.store(sig);
    const int fd = g_shutdownEventFd.load();
    if (fd < 0) {
        return;
    }
    const std::uint64_t one = 1;
    // eventfd write is atomic — we don't care about EAGAIN because that
    // means the counter is already saturated, which is equivalent to
    // "shutdown already requested".
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

    // SIGPIPE: send() on a closed client returns EPIPE via errno, which
    // is what we handle. The default action would kill Munin — hard no.
    struct ::sigaction sp{};
    sp.sa_handler = SIG_IGN;
    ::sigemptyset(&sp.sa_mask);
    if (::sigaction(SIGPIPE, &sp, nullptr) < 0) return false;
    return true;
}

// Ensure the parent directory of the socket path exists. `mkdir -p`
// semantics with mode 0755. Idempotent. Returns false on any error
// that is not EEXIST.
bool ensureParentDir(const std::string& path) noexcept {
    const auto slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0) {
        return true;  // path in cwd or root — nothing to create
    }
    const std::string dir = path.substr(0, slash);
    // Walk components from the leftmost /.
    std::string acc;
    acc.reserve(dir.size());
    for (std::size_t i = 0; i < dir.size(); ++i) {
        acc.push_back(dir[i]);
        if (dir[i] != '/' && i + 1 != dir.size()) {
            continue;
        }
        if (acc == "/") continue;
        if (::mkdir(acc.c_str(), 0755) < 0 && errno != EEXIST) {
            return false;
        }
    }
    return true;
}

} // namespace

int Daemon::run(const CliOptions& opts) noexcept {
    // ---- config -------------------------------------------------------
    ::mimirmind::core::config::Config cfg;
    try {
        if (opts.configPath.empty()) {
            std::fprintf(stderr,
                         "munin: --config is required (path to config.json)\n");
            return 2;
        }
        cfg = ::mimirmind::core::config::loadConfig(opts.configPath);
    } catch (const std::exception& x) {
        std::fprintf(stderr, "munin: load config failed: %s\n", x.what());
        return 2;
    }

    // CLI overrides for logging live on ServerSettings.log, which the
    // logger reads.
    if (!opts.logFile.empty())  cfg.server.log.file  = opts.logFile;
    if (!opts.logLevel.empty()) cfg.server.log.level = opts.logLevel;
    ::mimirmind::core::log::Log::initFromConfig(cfg.server.log);

    MM_LOG_INFO("munin",
                "starting: config='{}' pid={}",
                opts.configPath, static_cast<int>(::getpid()));

    // ---- L0 + allocator ----------------------------------------------
    // Munin always uses Host-USM for its allocations so it can export
    // IPC handles to attached workers (see M-Munin ADR "L0-IPC funktioniert
    // auf Meteor Lake nur mit zeMemAllocHost"). We ignore
    // selectUsmAllocKind's heuristic and force Host — attached workers
    // rely on this invariant.
    std::unique_ptr<::mimirmind::core::l0::L0Context>    l0;
    std::unique_ptr<::mimirmind::core::l0::UsmAllocator> allocator;
    try {
        const auto rt   = cfg.runtime;
        l0 = std::make_unique<::mimirmind::core::l0::L0Context>(
            rt.spvDir.value_or(std::string{}));
        allocator = std::make_unique<::mimirmind::core::l0::UsmAllocator>(
            *l0,
            rt.usmProbeTotalGib,
            ::mimirmind::core::l0::UsmAllocKind::Host);
    } catch (const std::exception& x) {
        MM_LOG_ERROR("munin", "L0 / UsmAllocator init failed: {}", x.what());
        return 3;
    }

    // ---- model store -------------------------------------------------
    std::unique_ptr<ModelStore> store;
    try {
        store = std::make_unique<ModelStore>(cfg, *allocator);
    } catch (const std::exception& x) {
        MM_LOG_ERROR("munin", "ModelStore init failed: {}", x.what());
        return 4;
    }
    allocator->logStats(::mimirmind::core::log::LogLevel::Info);

    // ---- socket path -------------------------------------------------
    std::string socketPath = opts.socketPath;
    if (socketPath.empty()) {
        // Default lives next to the systemd runtime dir on prod; on dev
        // machines /var/run may not exist without root. Compose files
        // set --socket explicitly for the bind-mounted path.
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
        SocketServer server{*store, *l0, socketPath};
        server.serve(evfd);
    } catch (const std::exception& x) {
        MM_LOG_ERROR("munin", "SocketServer failed: {}", x.what());
        exitCode = 7;
    }

    const int sig = g_lastSignal.load();
    if (sig != 0) {
        MM_LOG_INFO("munin",
                    "shutdown initiated by signal {} ({})",
                    sig,
                    sig == SIGINT  ? "SIGINT"  :
                    sig == SIGTERM ? "SIGTERM" : "?");
    }

    g_shutdownEventFd.store(-1);
    ::close(evfd);

    MM_LOG_INFO("munin", "exited (code={})", exitCode);
    return exitCode;
}

} // namespace mimirmind::munin