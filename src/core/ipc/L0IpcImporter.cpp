// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/ipc/L0IpcImporter.hpp"

#include "core/ipc/IpcImporter.hpp"

#include <level_zero/ze_api.h>

namespace mimirmind::core::ipc {

std::expected<void*, std::string>
L0IpcImporter::importChunk(std::span<const std::byte, 64> bytes,
                           int                             receivedFd) noexcept {
    return IpcImporter::importOne(_ctx, bytes, receivedFd);
}

void L0IpcImporter::closeChunk(void* ptr) noexcept {
    if (ptr != nullptr) {
        (void)::zeMemCloseIpcHandle(_ctx.context(), ptr);
    }
}

} // namespace mimirmind::core::ipc
