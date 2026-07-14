// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/l0/L0Context.hpp"

#include <array>
#include <cstddef>
#include <expected>
#include <string>

namespace mimirmind::core::ipc {

/**
 * Server-side (Munin) helper: extract a Level Zero IPC handle for one
 * USM allocation.
 *
 * The Intel L0 loader packs a dma_buf file-descriptor in the first
 * four bytes of the 64-byte handle blob on Linux; the surrounding bytes
 * carry flags the loader interprets on the receiving side (see
 * research/l0-ipc-host-only-meteor-lake.md for the byte-16=0x01 flag
 * that distinguishes Host-USM from Shared/Device on Meteor Lake).
 *
 * The caller sends the 64-byte blob as the frame payload and the fd
 * separately via SCM_RIGHTS. The receiver reconstructs the handle by
 * patching its first four bytes with the fd-number received via
 * SCM_RIGHTS (kernel dup'd, so the number differs between processes).
 *
 * The fd is NOT owned by the caller: it is a view into L0's internal
 * dma_buf table. `zeMemFree` cleans it up when the underlying
 * allocation is released. Do not close() it.
 */
struct IpcExporter {
    struct ExportedHandle {
        std::array<std::byte, 64> bytes{};
        int                       fd{-1};
    };

    [[nodiscard]] static std::expected<ExportedHandle, std::string>
    exportOne(::mimirmind::core::l0::L0Context& ctx, void* usmPtr) noexcept;
};

} // namespace mimirmind::core::ipc