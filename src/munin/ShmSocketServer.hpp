// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "munin/ShmAttachSession.hpp"
#include "munin/ShmModelStore.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mimirmind::munin {

/**
 * AF_UNIX SOCK_STREAM listen/accept loop that spawns one ShmAttachSession
 * (its own thread) per accepted connection — the shm analogue of
 * SocketServer (ADR 2026-08-14, step 4). Identical accept/poll/shutdown
 * mechanics; the only difference is it holds a ShmModelStore (memfd chunks)
 * and needs no L0Context, so it is pure host code in mimirmind_core_common.
 */
class ShmSocketServer {
public:
    ShmSocketServer(const ShmModelStore& store, std::string socketPath);
    ~ShmSocketServer();

    ShmSocketServer(const ShmSocketServer&)            = delete;
    ShmSocketServer& operator=(const ShmSocketServer&) = delete;
    ShmSocketServer(ShmSocketServer&&)                 = delete;
    ShmSocketServer& operator=(ShmSocketServer&&)      = delete;

    /// Bind + listen, then run the accept loop until stop() is called or
    /// `shutdownEventFd` becomes readable (-1 disables the external wake).
    /// Blocks in the caller's thread; returns after all sessions are joined.
    void serve(int shutdownEventFd);

    /// Signal the accept loop + all sessions to wind down. Idempotent.
    void stop() noexcept;

    struct SessionInfo {
        std::uint32_t sessionId{0};
        int           peerPid{0};
        std::string   modelId{};
    };
    [[nodiscard]] std::vector<SessionInfo> sessions() const;

private:
    struct SessionSlot {
        std::unique_ptr<ShmAttachSession> session;
        std::thread                       thread;
        int                               connFd{-1};
    };

    void closeListenFd() noexcept;

    const ShmModelStore& _store;
    std::string          _socketPath;

    int                        _listenFd{-1};
    std::atomic<bool>          _stopRequested{false};
    std::atomic<std::uint32_t> _nextSessionId{1};

    mutable std::mutex                        _sessionsMx;
    std::vector<std::unique_ptr<SessionSlot>> _sessions;
};

} // namespace mimirmind::munin
