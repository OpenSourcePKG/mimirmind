// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/ipc/IpcImporter.hpp"

#include <level_zero/ze_api.h>

#include <cstring>
#include <sstream>

namespace mimirmind::core::ipc {

std::expected<void*, std::string>
IpcImporter::importOne(::mimirmind::core::l0::L0Context& ctx,
                       std::span<const std::byte, 64>    handleBytes,
                       int                               receivedFd) noexcept {
    if (receivedFd < 0) {
        return std::unexpected(std::string{"IpcImporter: receivedFd is negative"});
    }

    ze_ipc_mem_handle_t patched{};
    static_assert(sizeof(patched.data) == 64,
                  "ze_ipc_mem_handle_t.data must be 64 bytes — L0 headers changed?");
    std::memcpy(patched.data, handleBytes.data(), sizeof(patched.data));

    // Patch the first four bytes (Linux dma_buf fd) with the fd we
    // received via SCM_RIGHTS — that number is only meaningful in this
    // process. The rest of the 64-byte blob carries flags the L0
    // loader parses to reconstruct the allocation's placement metadata.
    std::memcpy(patched.data, &receivedFd, sizeof(int));

    void* ptr = nullptr;
    const ze_result_t r = ::zeMemOpenIpcHandle(
        ctx.context(), ctx.device(), patched, /*flags=*/0, &ptr);
    if (r != ZE_RESULT_SUCCESS) {
        std::ostringstream os;
        os << "IpcImporter: zeMemOpenIpcHandle(fd=" << receivedFd
           << ") -> " << ::mimirmind::core::l0::L0Context::resultToString(r)
           << " (0x" << std::hex << static_cast<unsigned>(r) << ")";
        return std::unexpected(os.str());
    }
    if (ptr == nullptr) {
        return std::unexpected(std::string{
            "IpcImporter: zeMemOpenIpcHandle returned SUCCESS but ptr is null"});
    }
    return ptr;
}

std::expected<std::vector<void*>, std::string>
IpcImporter::openChunks(
    ::mimirmind::core::l0::L0Context&                     ctx,
    std::span<const std::array<std::byte, 64>>            handleBlobs,
    std::span<const int>                                  receivedFds) noexcept {
    if (handleBlobs.size() != receivedFds.size()) {
        std::ostringstream os;
        os << "IpcImporter::openChunks: size mismatch — "
           << handleBlobs.size() << " blob(s) vs " << receivedFds.size()
           << " fd(s)";
        return std::unexpected(os.str());
    }

    std::vector<void*> bases;
    bases.reserve(handleBlobs.size());

    for (std::size_t i = 0; i < handleBlobs.size(); ++i) {
        std::span<const std::byte, 64> blob{handleBlobs[i].data(),
                                            handleBlobs[i].size()};
        auto p = importOne(ctx, blob, receivedFds[i]);
        if (!p) {
            // Rollback: release every L0 mapping we already opened. The
            // fd for chunk i was NOT taken by importOne (it failed), and
            // fds i+1..N-1 were never touched — caller closes them all.
            for (void* prev : bases) {
                (void)::zeMemCloseIpcHandle(ctx.context(), prev);
            }
            std::ostringstream os;
            os << "IpcImporter::openChunks: chunk[" << i << "] failed: "
               << p.error();
            return std::unexpected(os.str());
        }
        bases.push_back(*p);
    }
    return bases;
}

} // namespace mimirmind::core::ipc