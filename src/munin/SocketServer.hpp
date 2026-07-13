#pragma once

#include "core/l0/L0Context.hpp"
#include "munin/AttachSession.hpp"
#include "munin/ModelStore.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace mimirmind::munin {

/**
 * AF_UNIX SOCK_STREAM listen/accept loop that spawns one
 * `AttachSession` (in its own std::thread) per accepted connection.
 *
 * The socket path is created on `start()` and unlinked on `stop()`.
 * If a stale file already exists at that path and it is a socket AND
 * nobody is listening (connect fails with ECONNREFUSED), we unlink it
 * and retry. Anything else is refused — we do not clobber unknown
 * files.
 *
 * Concurrency:
 *   - The listen fd is owned by the accept thread only.
 *   - `_sessions` and `_workers` are guarded by `_sessionsMx`.
 *   - `stop()` closes the listen fd (unblocks accept) and closes every
 *     active connection fd (unblocks the per-session recv), then joins
 *     all worker threads.
 */
class SocketServer {
public:
    SocketServer(const ModelStore&                  store,
                 ::mimirmind::core::l0::L0Context&  l0,
                 std::string                        socketPath);

    ~SocketServer();

    SocketServer(const SocketServer&)            = delete;
    SocketServer& operator=(const SocketServer&) = delete;
    SocketServer(SocketServer&&)                 = delete;
    SocketServer& operator=(SocketServer&&)      = delete;

    /**
     * Bind + listen, then run the accept loop until `stop()` is called
     * or `shutdownEventFd` becomes readable. `shutdownEventFd` is an
     * eventfd/pipe fd owned by the caller; passing -1 disables the
     * external wake-up (only `stop()` from another thread will exit
     * the loop).
     *
     * Blocks in the caller's thread. Returns after the accept loop has
     * exited and all pending sessions have been joined.
     */
    void serve(int shutdownEventFd);

    /**
     * Signal the accept loop and all sessions to wind down. Safe to
     * call from a signal handler? No — it takes a mutex and joins
     * threads. Call from a normal thread (typically the daemon's
     * signal-dispatch thread). Idempotent.
     */
    void stop() noexcept;

    /// Snapshot of currently-attached sessions for diagnostics.
    struct SessionInfo {
        std::uint32_t sessionId{0};
        int           peerPid{0};
        std::string   modelId{};
    };
    [[nodiscard]] std::vector<SessionInfo> sessions() const;

private:
    /// One accepted connection: session object + the thread driving it.
    struct SessionSlot {
        std::unique_ptr<AttachSession> session;
        std::thread                    thread;
        int                            connFd{-1};   // duplicate of session's fd for shutdown
    };

    void closeListenFd() noexcept;
    void reapFinishedSessions() noexcept;

    const ModelStore&                    _store;
    ::mimirmind::core::l0::L0Context&    _l0;
    std::string                          _socketPath;

    int                                  _listenFd{-1};
    std::atomic<bool>                    _stopRequested{false};
    std::atomic<std::uint32_t>           _nextSessionId{1};

    mutable std::mutex                   _sessionsMx;
    std::vector<std::unique_ptr<SessionSlot>> _sessions;
};

} // namespace mimirmind::munin