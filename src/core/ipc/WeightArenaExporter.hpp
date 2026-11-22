// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/ipc/WeightArenaManifest.hpp"

#include <atomic>
#include <expected>
#include <string>
#include <thread>

namespace mimirmind::core::ipc {

/**
 * 7.2.3 — serves an engine's weight arena to attachers on a Unix socket.
 *
 * Handshake: the client sends one frame (`{"op":"attach",...}`), the server
 * answers with one frame whose payload is the manifest JSON and whose
 * SCM_RIGHTS fd is a READ-ONLY descriptor of the arena memfd (reopened via
 * /proc/self/fd with O_RDONLY), so an attacher cannot map it writable. Live
 * weight writes from outside the engine are 7.7 and need the quiesce API.
 *
 * One background accept thread; `stop()` / the destructor joins it and
 * unlinks the socket path.
 */
class WeightArenaExporter {
public:
    WeightArenaExporter() = default;
    ~WeightArenaExporter();

    WeightArenaExporter(const WeightArenaExporter&)            = delete;
    WeightArenaExporter& operator=(const WeightArenaExporter&) = delete;

    /// Bind `socketPath` (a stale socket file is replaced) and start serving.
    [[nodiscard]] std::expected<void, std::string>
    start(const std::string& socketPath, int arenaFd, WeightArenaManifest manifest);

    void stop() noexcept;

    [[nodiscard]] std::uint64_t attachCount() const noexcept { return _attaches.load(); }

private:
    void run() noexcept;

    std::string               _path;
    int                       _listenFd{-1};
    int                       _roFd{-1};
    std::string               _manifestJson;
    std::thread               _thread;
    std::atomic<bool>         _stop{false};
    std::atomic<std::uint64_t> _attaches{0};
};

/// Attacher side: connect to `socketPath`, run the handshake, return the
/// manifest plus the received arena fd (caller owns it).
struct WeightArenaHandle {
    WeightArenaManifest manifest;
    int                 fd{-1};
};

[[nodiscard]] std::expected<WeightArenaHandle, std::string>
fetchWeightArena(const std::string& socketPath, const std::string& clientName) noexcept;

} // namespace mimirmind::core::ipc
