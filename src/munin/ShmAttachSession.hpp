// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "munin/ShmModelStore.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>
#include <sys/types.h>

namespace mimirmind::munin {

/**
 * One accepted worker connection for the CUDA/GB10 (POSIX-shm) Munin — the
 * shm analogue of AttachSession (ADR 2026-08-14, step 4). Owns the socket fd
 * for its lifetime; `run()` services exactly one request envelope, then on
 * `attach` keeps the socket open so Munin observes worker disconnect as an
 * implicit detach.
 *
 * Speaks the identical wire (RequestEnvelope -> healthz JSON, or manifest +
 * one HANDLE frame per chunk) but exports each chunk with ShmIpcExporter
 * (the borrowed memfd + a 64-byte length payload) instead of an L0 IPC
 * handle. Needs no L0Context — the memfds live in the ShmModelStore. Pure
 * host code, builds in mimirmind_core_common.
 *
 * Not copy/move-constructible; heap-owned behind unique_ptr, driven from its
 * own thread by the socket server.
 */
class ShmAttachSession {
public:
    ShmAttachSession(int                   fd,
                     pid_t                 peerPid,
                     std::uint32_t         sessionId,
                     const ShmModelStore&  store);

    ~ShmAttachSession();

    ShmAttachSession(const ShmAttachSession&)            = delete;
    ShmAttachSession& operator=(const ShmAttachSession&) = delete;
    ShmAttachSession(ShmAttachSession&&)                 = delete;
    ShmAttachSession& operator=(ShmAttachSession&&)      = delete;

    /// Handle one request envelope, then either return (healthz / error) or
    /// block until peer close (attach). Invoke exactly once per instance.
    void run() noexcept;

    /// Ask the session to wind down at the next syscall boundary. Idempotent.
    void requestStop() noexcept { _stopRequested.store(true); }

    [[nodiscard]] std::uint32_t sessionId() const noexcept { return _sessionId; }
    [[nodiscard]] pid_t         peerPid()   const noexcept { return _peerPid; }
    [[nodiscard]] std::string   attachedModelId() const;

private:
    void sendErrorAndClose(std::string_view msg) noexcept;
    bool handleHealthz() noexcept;
    bool handleAttach(std::string_view modelId) noexcept;
    void waitForPeerClose() noexcept;

    int                  _fd;
    pid_t                _peerPid;
    std::uint32_t        _sessionId;
    const ShmModelStore& _store;
    std::atomic<bool>    _stopRequested{false};
    std::string          _attachedModelId{};
};

} // namespace mimirmind::munin
