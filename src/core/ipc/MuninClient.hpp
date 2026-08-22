// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/ipc/IpcTransport.hpp"
#include "core/ipc/TensorManifest.hpp"
#include "core/ipc/WireOps.hpp"

#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::ipc {

/**
 * Worker-side attach client for the M-Munin wire protocol — backend-neutral
 * (M-Munin 1b-tail). Represents one connection to a running Munin daemon and
 * speaks the wire (UnixSocketFrame framing, RequestEnvelope, TensorManifest
 * v2, N HANDLE frames of 64-byte payload + one SCM_RIGHTS fd). The one
 * transport-specific step — turning a received handle into a local pointer —
 * is delegated to an `IpcImporterBackend`: L0IpcImporter (zeMemOpenIpcHandle,
 * Xe-LPG) or ShmIpcImporter (mmap the memfd, GB10). The resulting chunkBases
 * feed `WeightsMap::fromAttachedChunked` (GGUF) or, for NVFP4, the shard
 * reconstruction in `InferenceEngine::loadModelAttachedNvfp4`.
 *
 * The importer is a reference — it MUST outlive this client (the imported
 * mappings stay valid for the worker's run). Pure host code in
 * mimirmind_core_common; not thread-safe, one per worker, blocking I/O.
 */
class MuninClient {
public:
    explicit MuninClient(IpcImporterBackend& importer) noexcept
        : _importer{importer} {}
    ~MuninClient();

    MuninClient(const MuninClient&)            = delete;
    MuninClient& operator=(const MuninClient&) = delete;
    MuninClient(MuninClient&&)                 = delete;
    MuninClient& operator=(MuninClient&&)      = delete;

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
    /// takes ownership of `fd`; on failure the caller still owns it. Exposed
    /// so the flow can be driven over a socketpair in tests.
    [[nodiscard]] std::expected<AttachResult, std::string>
    attachOnConnectedFd(int fd, std::string_view modelId) noexcept;

    [[nodiscard]] bool isAttached() const noexcept { return _sessionFd >= 0; }

    /// Close the session socket (idempotent). Does NOT release the imported
    /// mappings — those are owned by the importer and live for the worker's
    /// run.
    void detach() noexcept;

private:
    IpcImporterBackend& _importer;
    int                 _sessionFd{-1};
};

} // namespace mimirmind::core::ipc
