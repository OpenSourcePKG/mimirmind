// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>

namespace mimirmind::core::ipc {

/**
 * Backend-neutral IPC transport seam for the M-Munin daemon.
 *
 * Munin holds a model resident and exports it to an attached worker as a
 * sequence of chunks (M-Munin.1a: one HANDLE frame per chunk, not per
 * tensor). The wire format — TensorManifest + N HANDLE frames, each a
 * 64-byte payload plus one SCM_RIGHTS fd — is already backend-agnostic;
 * the ONLY backend-specific step is turning a resident chunk into a
 * transportable handle (server) and turning a received handle back into a
 * local pointer (worker). This header abstracts exactly those two steps so
 * the same AttachSession / MuninClient code drives either backend:
 *
 *   - L0  (Meteor Lake / Xe-LPG): zeMemGetIpcHandle / zeMemOpenIpcHandle
 *     over USM-host allocations. See L0IpcExporter / L0IpcImporter.
 *   - shm (DGX Spark / GB10):     POSIX memfd + mmap. The worker maps the
 *     memfd and hands the host pointer straight to CUDA kernels, relying
 *     on GB10 pageableMemAccess + hostPageTables (verified end-to-end,
 *     tools/cuda-ipc-testrig --kind shm). See ShmIpcExporter /
 *     ShmIpcImporter.
 *
 * ADR: decisions/2026-08-14-munin-cuda-port-shm.md (step 1b).
 */

/**
 * One chunk's transportable IPC handle. `bytes` is the 64-byte HANDLE-frame
 * payload; `fd` travels separately via SCM_RIGHTS. The meaning of `bytes`
 * is backend-defined:
 *   - L0  : the 64-byte ze_ipc_mem_handle blob. `fd` is a BORROWED view
 *           into L0's dma_buf table (do not close; zeMemFree releases it).
 *   - shm : first 8 bytes = little-endian mmap length; the rest is zero.
 *           `fd` is the memfd, BORROWED from the server-side allocator that
 *           keeps the model resident (do not close on the server).
 *
 * In both backends the exporter returns a borrowed fd: SCM_RIGHTS dups it
 * into the receiver, and the server keeps its own copy for the model's
 * lifetime, so AttachSession sends the fd and never closes it.
 */
struct IpcHandle {
    std::array<std::byte, 64> bytes{};
    int                       fd{-1};
};

/**
 * Server side (Munin): turn one resident model chunk into an IpcHandle.
 *
 * `exportChunk` is called once per chunk in manifest order. `base` is the
 * chunk's server-side pointer (used by the L0 backend to look up its IPC
 * handle); `usedBytes` is the chunk's populated footprint (carried for
 * sanity, unused by L0). A backend that is indexed by chunk rather than by
 * pointer (shm) uses `index`. The returned fd is borrowed — the caller
 * transmits it and must not close it.
 */
class IpcExporterBackend {
public:
    virtual ~IpcExporterBackend() = default;

    [[nodiscard]] virtual std::expected<IpcHandle, std::string>
    exportChunk(std::uint32_t index, void* base, std::uint64_t usedBytes) noexcept = 0;
};

/**
 * Client side (worker): import a received handle into this address space.
 *
 * `importChunk` consumes `receivedFd` on success (the backend takes over
 * ownership — for shm the mapping survives closing the fd; for L0 the
 * loader adopts it). On failure the caller still owns `receivedFd`. The
 * importer remembers whatever it needs to release each pointer later, so
 * `closeChunk` takes only the pointer.
 */
class IpcImporterBackend {
public:
    virtual ~IpcImporterBackend() = default;

    [[nodiscard]] virtual std::expected<void*, std::string>
    importChunk(std::span<const std::byte, 64> bytes, int receivedFd) noexcept = 0;

    virtual void closeChunk(void* ptr) noexcept = 0;
};

} // namespace mimirmind::core::ipc
