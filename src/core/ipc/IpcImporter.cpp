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

} // namespace mimirmind::core::ipc