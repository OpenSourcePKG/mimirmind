// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace mimirmind::core::ipc {

/**
 * One decoded frame received on a Unix stream socket. `payload` holds
 * the raw bytes; `fds` holds any file descriptors that arrived on the
 * same message via SCM_RIGHTS.
 *
 * Ownership: the receiver owns the fds and must close() them when done
 * (or hand them to code that will, e.g. zeMemOpenIpcHandle which takes
 * over the fd).
 */
struct Frame {
    std::vector<std::byte> payload{};
    std::vector<int>       fds{};
};

/**
 * Length-prefixed framing on a SOCK_STREAM Unix socket, optionally with
 * SCM_RIGHTS file descriptors attached.
 *
 * Wire format per frame:
 *   [4-byte little-endian uint32 length][N bytes payload]
 *
 * The SCM_RIGHTS control message rides along with the 4-byte length
 * prefix (any msg with >= 1 byte of data can carry ancillary control).
 * That decouples the payload transfer from the fd handoff; the payload
 * bytes then stream without ancillary data.
 *
 * Blocking mode expected. The functions loop internally until the full
 * frame is transferred or the peer closes. Not thread-safe on a shared
 * socket — one send and one recv may run concurrently on the same fd,
 * but two concurrent senders or two concurrent receivers will interleave.
 *
 * All methods return std::expected. The `std::string` error carries a
 * short human-readable reason for logging; no stack trace, no errno
 * translation. The M-Munin daemon and the attach-worker convert it into
 * appropriate log lines and connection-teardown behaviour.
 */
struct UnixSocketFrame {
    /// Maximum fds per single frame (Linux kernel SCM_MAX_FD is 253).
    static constexpr std::size_t kMaxFdsPerFrame = 253;

    /// Default safety cap on payload size for recv() to prevent a
    /// malicious peer from allocating unbounded memory.
    static constexpr std::size_t kDefaultMaxPayload = 64ULL * 1024 * 1024;

    [[nodiscard]] static std::expected<void, std::string>
    send(int sock,
         std::span<const std::byte> payload,
         std::span<const int>       fds = {}) noexcept;

    [[nodiscard]] static std::expected<Frame, std::string>
    recv(int sock,
         std::size_t maxPayloadBytes = kDefaultMaxPayload) noexcept;
};

} // namespace mimirmind::core::ipc