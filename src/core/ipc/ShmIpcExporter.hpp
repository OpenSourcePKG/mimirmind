// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/ipc/IpcTransport.hpp"

#include <cstdint>
#include <span>

namespace mimirmind::core::ipc {

/**
 * POSIX-shm (memfd) implementation of IpcExporterBackend — the chosen
 * M-Munin.CUDA path (ADR 2026-08-14, step 1b).
 *
 * Unlike the L0 backend, an shm chunk cannot be turned into a handle from
 * an arbitrary pointer: the transportable identity IS the memfd that backs
 * the chunk. So this exporter is constructed with the per-chunk memfds and
 * their mmap lengths — owned by the server-side shm allocator (the future
 * ShmChunkAllocator / shm ModelStore, step 2), which keeps them open for
 * the model's whole lifetime. `exportChunk(i)` just returns memfd[i] as a
 * BORROWED fd plus a 64-byte payload encoding the mmap length; SCM_RIGHTS
 * dups the fd into the worker, and the server keeps its own copy.
 *
 * Pure host code — no CUDA here. The memfd is plain host RAM; the worker
 * maps it and hands the pointer straight to CUDA kernels (GB10
 * pageableMemAccess + hostPageTables, verified: tools/cuda-ipc-testrig
 * --kind shm). That is why this class lives in mimirmind_core_common and
 * needs no GPU SDK to build or test.
 *
 * The two spans are VIEWS: their backing arrays must outlive this exporter.
 */
class ShmIpcExporter final : public IpcExporterBackend {
public:
    ShmIpcExporter(std::span<const int>            chunkMemfds,
                   std::span<const std::uint64_t>  chunkMapBytes) noexcept;

    [[nodiscard]] std::expected<IpcHandle, std::string>
    exportChunk(std::uint32_t index, void* base,
                std::uint64_t usedBytes) noexcept override;

private:
    std::span<const int>           _memfds;
    std::span<const std::uint64_t> _mapBytes;
};

} // namespace mimirmind::core::ipc
