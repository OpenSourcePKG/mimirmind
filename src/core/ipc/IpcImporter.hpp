// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/l0/L0Context.hpp"

#include <array>
#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace mimirmind::core::ipc {

/**
 * Client-side (attached mimirmind worker) helper: import a Level Zero
 * IPC handle that Munin sent over the socket.
 *
 * Munin sends the 64-byte handle blob as frame payload and the dma_buf
 * fd separately via SCM_RIGHTS. The kernel dup'd the fd when it crossed
 * the socket, so the fd-number embedded in the blob (from Munin's file
 * table) is stale in this process. `importOne` patches the blob's first
 * four bytes with the fd we actually received, then calls
 * `zeMemOpenIpcHandle` to obtain a user-space pointer.
 *
 * Ownership: on success, the receiver has taken over the fd — do NOT
 * close() it. `zeMemCloseIpcHandle(ctx, ptr)` releases it later.
 * On failure, the caller must close(receivedFd) itself.
 */
struct IpcImporter {
    [[nodiscard]] static std::expected<void*, std::string>
    importOne(::mimirmind::core::l0::L0Context& ctx,
              std::span<const std::byte, 64>    handleBytes,
              int                               receivedFd) noexcept;

    /**
     * Import an ordered sequence of chunk IPC handles in one shot. Used
     * by the M-Munin.1a chunk-layout attach path — the worker walks
     * `manifest.chunks` and calls this with the collected 64-byte blobs
     * and their matching SCM_RIGHTS fds.
     *
     * `handleBlobs.size()` must equal `receivedFds.size()`; a mismatch
     * is a caller bug and fails without touching any handles.
     *
     * Ownership rules — all-or-nothing to keep the caller path simple:
     *   - On success, returns N pointers (chunk bases). Every fd is
     *     consumed by L0 and the caller must NOT close() them; each
     *     mapping is released later via `zeMemCloseIpcHandle(ctx, ptr)`.
     *   - On failure at index K, every ptr already opened for
     *     chunks 0..K-1 is rolled back via `zeMemCloseIpcHandle`, so
     *     the L0 side leaks nothing. The fds themselves — for every
     *     chunk 0..N-1 — remain owned by the caller (`receivedFds[K]`
     *     was not consumed since `importOne` failed; K+1..N-1 were
     *     never touched). The caller closes them all.
     */
    [[nodiscard]] static std::expected<std::vector<void*>, std::string>
    openChunks(::mimirmind::core::l0::L0Context&                     ctx,
               std::span<const std::array<std::byte, 64>>            handleBlobs,
               std::span<const int>                                  receivedFds) noexcept;
};

} // namespace mimirmind::core::ipc