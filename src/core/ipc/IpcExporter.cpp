// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/ipc/IpcExporter.hpp"

#include <level_zero/ze_api.h>

#include <cstring>
#include <sstream>

namespace mimirmind::core::ipc {

std::expected<IpcExporter::ExportedHandle, std::string>
IpcExporter::exportOne(::mimirmind::core::l0::L0Context& ctx, void* usmPtr) noexcept {
    if (usmPtr == nullptr) {
        return std::unexpected(std::string{"IpcExporter: usmPtr is null"});
    }

    ze_ipc_mem_handle_t raw{};
    const ze_result_t r = ::zeMemGetIpcHandle(ctx.context(), usmPtr, &raw);
    if (r != ZE_RESULT_SUCCESS) {
        std::ostringstream os;
        os << "IpcExporter: zeMemGetIpcHandle(ptr=" << usmPtr
           << ") -> " << ::mimirmind::core::l0::L0Context::resultToString(r)
           << " (0x" << std::hex << static_cast<unsigned>(r) << ")";
        return std::unexpected(os.str());
    }

    static_assert(sizeof(raw.data) == 64,
                  "ze_ipc_mem_handle_t.data must be 64 bytes — L0 headers changed?");

    ExportedHandle out{};
    std::memcpy(out.bytes.data(), raw.data, sizeof(raw.data));
    // First 4 bytes carry the dma_buf fd on Linux (verified in
    // tools/l0-ipc-testrig on Meteor Lake). Extract for SCM_RIGHTS
    // transmission — the receiver will patch it back in after the
    // kernel dup.
    int fd = -1;
    std::memcpy(&fd, raw.data, sizeof(int));
    out.fd = fd;
    return out;
}

} // namespace mimirmind::core::ipc