// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/ipc/IpcTransport.hpp"

#include <cstddef>
#include <span>
#include <unordered_map>

namespace mimirmind::core::ipc {

/**
 * POSIX-shm (memfd) implementation of IpcImporterBackend — the worker side
 * of the chosen M-Munin.CUDA path (ADR 2026-08-14, step 1b).
 *
 * `importChunk` reads the mmap length from the 64-byte payload and maps the
 * received memfd (MAP_SHARED) into this process. The resulting host pointer
 * is handed straight to CUDA kernels by the caller — GB10 pageableMemAccess
 * + hostPageTables let the SMs dereference it directly, no cudaHostRegister
 * (verified: tools/cuda-ipc-testrig --kind shm). The mapping survives
 * closing the fd, so `importChunk` closes `receivedFd` after a successful
 * mmap and remembers `ptr -> length` for `closeChunk` / teardown.
 *
 * Pure host code — no CUDA SDK needed to build (lives in
 * mimirmind_core_common). Not thread-safe; one per worker, mirroring
 * MuninClient's single-connection lifetime.
 */
class ShmIpcImporter final : public IpcImporterBackend {
public:
    ShmIpcImporter() noexcept = default;
    ~ShmIpcImporter() override;

    ShmIpcImporter(const ShmIpcImporter&)            = delete;
    ShmIpcImporter& operator=(const ShmIpcImporter&) = delete;
    ShmIpcImporter(ShmIpcImporter&&)                 = delete;
    ShmIpcImporter& operator=(ShmIpcImporter&&)      = delete;

    [[nodiscard]] std::expected<void*, std::string>
    importChunk(std::span<const std::byte, 64> bytes,
                int                             receivedFd) noexcept override;

    void closeChunk(void* ptr) noexcept override;

private:
    // Every live mapping we own, so closeChunk / the destructor can munmap
    // with the correct length without the caller tracking sizes.
    std::unordered_map<void*, std::size_t> _maps;
};

} // namespace mimirmind::core::ipc
