// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/ipc/ShmIpcImporter.hpp"
#include "core/ipc/TensorManifest.hpp"
#include "core/ipc/WireOps.hpp"

#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::ipc {

/**
 * Worker-side attach client for the M-Munin.CUDA (POSIX-shm) transport —
 * the shm analogue of MuninClient (ADR 2026-08-14, step 3).
 *
 * Speaks the exact same backend-neutral wire (UnixSocketFrame framing,
 * RequestEnvelope, TensorManifest v2, N HANDLE frames each 64-byte payload
 * + one SCM_RIGHTS fd) but imports each chunk with ShmIpcImporter (mmap the
 * received memfd) instead of L0's zeMemOpenIpcHandle. The returned
 * `chunkBases` feed the SAME backend-neutral resolver,
 * `WeightsMap::fromAttachedChunked(manifest, chunkBases)` — on GB10 those
 * host pointers go straight to CUDA kernels (pageableMemAccess +
 * hostPageTables, verified: tools/cuda-ipc-testrig --kind shm).
 *
 * Pure host code — no CUDA/L0 SDK — so it lives in mimirmind_core_common.
 * (The wire logic is duplicated from MuninClient for now; unifying the two
 * clients behind an IpcImporterBackend& is the 1b-tail refactor, done on an
 * L0 box.)
 *
 * The socket stays open for the worker's whole run so Munin sees the peer
 * close as an implicit detach; the imported mmaps stay valid until this
 * client is destroyed. Not thread-safe; construct one per worker.
 */
class ShmMuninClient {
public:
    ShmMuninClient() noexcept = default;
    ~ShmMuninClient();

    ShmMuninClient(const ShmMuninClient&)            = delete;
    ShmMuninClient& operator=(const ShmMuninClient&) = delete;
    ShmMuninClient(ShmMuninClient&&)                 = delete;
    ShmMuninClient& operator=(ShmMuninClient&&)      = delete;

    [[nodiscard]] static std::expected<HealthzResponse, std::string>
    healthz(std::string_view socketPath) noexcept;

    struct AttachResult {
        TensorManifest     manifest;
        std::vector<void*> chunkBases;  // one per manifest.chunks, in order
    };

    /// Connect to Munin at `socketPath` and run the attach flow for
    /// `modelId`. On success the session socket stays open (see detach()).
    [[nodiscard]] std::expected<AttachResult, std::string>
    attach(std::string_view socketPath, std::string_view modelId) noexcept;

    /// Run the attach protocol on an ALREADY-CONNECTED socket. On success
    /// takes ownership of `fd` (kept as the session fd); on failure the
    /// caller still owns `fd`. Exposed so the flow can be driven over a
    /// socketpair in tests without a filesystem socket.
    [[nodiscard]] std::expected<AttachResult, std::string>
    attachOnConnectedFd(int fd, std::string_view modelId) noexcept;

    [[nodiscard]] bool isAttached() const noexcept { return _sessionFd >= 0; }

    /// Close the session socket (idempotent). Does NOT release the imported
    /// mmaps — those stay valid for the worker's run and are freed when the
    /// client is destroyed.
    void detach() noexcept;

private:
    ShmIpcImporter _importer;
    int            _sessionFd{-1};
};

} // namespace mimirmind::core::ipc
