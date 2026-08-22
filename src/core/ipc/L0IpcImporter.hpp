// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/l0/L0Context.hpp"
#include "core/ipc/IpcTransport.hpp"

namespace mimirmind::core::ipc {

/**
 * Level Zero implementation of IpcImporterBackend (M-Munin 1b-tail). Adapts
 * the existing IpcImporter::importOne / zeMemCloseIpcHandle behind the
 * backend-neutral seam so the one MuninClient wire implementation drives
 * either transport (L0 IPC handles on Xe-LPG, POSIX-shm on GB10 via
 * ShmIpcImporter). Lives in mimirmind_core_l0 — it holds an L0Context.
 */
class L0IpcImporter final : public IpcImporterBackend {
public:
    explicit L0IpcImporter(::mimirmind::core::l0::L0Context& ctx) noexcept
        : _ctx{ctx} {}

    [[nodiscard]] std::expected<void*, std::string>
    importChunk(std::span<const std::byte, 64> bytes,
                int                             receivedFd) noexcept override;

    void closeChunk(void* ptr) noexcept override;

private:
    ::mimirmind::core::l0::L0Context& _ctx;
};

} // namespace mimirmind::core::ipc
