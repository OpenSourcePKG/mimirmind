#pragma once

#include "core/l0/L0Context.hpp"

#include <cstddef>
#include <expected>
#include <span>
#include <string>

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
};

} // namespace mimirmind::core::ipc