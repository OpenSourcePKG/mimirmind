// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/ipc/ShmIpcImporter.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sstream>

namespace mimirmind::core::ipc {

ShmIpcImporter::~ShmIpcImporter() {
    // Safety net: release any mapping the caller did not close explicitly.
    for (const auto& [ptr, len] : _maps) {
        ::munmap(ptr, len);
    }
    _maps.clear();
}

std::expected<void*, std::string>
ShmIpcImporter::importChunk(std::span<const std::byte, 64> bytes,
                            int                             receivedFd) noexcept {
    if (receivedFd < 0) {
        return std::unexpected(std::string{"ShmIpcImporter: receivedFd is negative"});
    }

    // Decode the mmap length the exporter packed into the first 8 bytes.
    std::uint64_t mapLen = 0;
    std::memcpy(&mapLen, bytes.data(), sizeof(mapLen));
    if (mapLen == 0) {
        return std::unexpected(std::string{
            "ShmIpcImporter: handle declares a zero-length mapping"});
    }

    void* ptr = ::mmap(nullptr, static_cast<std::size_t>(mapLen),
                       PROT_READ | PROT_WRITE, MAP_SHARED, receivedFd, 0);
    if (ptr == MAP_FAILED) {
        const int e = errno;
        std::ostringstream os;
        os << "ShmIpcImporter: mmap(len=" << mapLen << ", fd=" << receivedFd
           << ") failed: " << std::strerror(e) << " (errno=" << e << ")";
        return std::unexpected(os.str());
    }

    // The mapping keeps the memfd alive; we no longer need the fd number.
    // Take ownership per the interface contract and close it.
    ::close(receivedFd);

    _maps.emplace(ptr, static_cast<std::size_t>(mapLen));
    return ptr;
}

void ShmIpcImporter::closeChunk(void* ptr) noexcept {
    if (ptr == nullptr) {
        return;
    }
    const auto it = _maps.find(ptr);
    if (it == _maps.end()) {
        return;  // not ours (or already closed) — no-op, never munmap blind
    }
    ::munmap(it->first, it->second);
    _maps.erase(it);
}

} // namespace mimirmind::core::ipc
