#pragma once

#include "core/l0/L0Context.hpp"
#include "munin/ModelStore.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>
#include <sys/types.h>

namespace mimirmind::munin {

/**
 * One accepted worker connection. Owns the socket fd for its lifetime;
 * `run()` is a blocking loop that services exactly one request envelope
 * and then, on `attach`, keeps the socket open so Munin can observe
 * worker disconnect as an implicit detach (that is Tier-1's cleanup
 * signal — see ADR).
 *
 * Constructed by `SocketServer` after `accept()`; typically driven from
 * its own std::thread. Cancellation from the daemon side: SocketServer
 * ::close(fd) on shutdown wakes any blocking recv with a peer-closed
 * result, `run()` returns and the thread joins.
 *
 * Not copy- or move-constructible — the fd ownership makes both
 * error-prone. A session on the heap behind unique_ptr is the intended
 * pattern.
 */
class AttachSession {
public:
    AttachSession(int                                        fd,
                  pid_t                                      peerPid,
                  std::uint32_t                              sessionId,
                  const ModelStore&                          store,
                  ::mimirmind::core::l0::L0Context&          l0);

    ~AttachSession();

    AttachSession(const AttachSession&)            = delete;
    AttachSession& operator=(const AttachSession&) = delete;
    AttachSession(AttachSession&&)                 = delete;
    AttachSession& operator=(AttachSession&&)      = delete;

    /**
     * Handle one request envelope, then either return (healthz / error)
     * or block reading from the socket until peer close (attach). Safe
     * to invoke exactly once per instance.
     */
    void run() noexcept;

    /// Ask the session to wind down at the next syscall boundary. Called
    /// by SocketServer on daemon shutdown. Idempotent.
    void requestStop() noexcept { _stopRequested.store(true); }

    [[nodiscard]] std::uint32_t sessionId() const noexcept { return _sessionId; }
    [[nodiscard]] pid_t         peerPid()   const noexcept { return _peerPid; }

    /// Currently-selected model id, empty until attach succeeded. Used by
    /// SocketServer for diagnostics.
    [[nodiscard]] std::string attachedModelId() const;

private:
    // Result path helpers. All log on failure; the return value is
    // "should we keep this session alive?" — false means we're done.
    void sendErrorAndClose(std::string_view msg) noexcept;
    bool handleHealthz() noexcept;
    bool handleAttach(std::string_view modelId) noexcept;

    // Blocks reading from _fd until the peer closes or a socket error
    // occurs. Used after a successful attach so Munin holds the connection
    // open for the worker's lifetime.
    void waitForPeerClose() noexcept;

    int                                 _fd;
    pid_t                               _peerPid;
    std::uint32_t                       _sessionId;
    const ModelStore&                   _store;
    ::mimirmind::core::l0::L0Context&   _l0;
    std::atomic<bool>                   _stopRequested{false};
    // Guarded implicitly by "only run() writes it, other threads read
    // it after run() has returned or via requestStop() semantics" —
    // std::string atomic ops would be overkill for a diagnostic field.
    std::string                         _attachedModelId{};
};

} // namespace mimirmind::munin