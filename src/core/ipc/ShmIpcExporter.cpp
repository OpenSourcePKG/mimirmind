// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/ipc/ShmIpcExporter.hpp"

#include <cstring>
#include <sstream>

namespace mimirmind::core::ipc {

ShmIpcExporter::ShmIpcExporter(std::span<const int>           chunkMemfds,
                               std::span<const std::uint64_t> chunkMapBytes) noexcept
    : _memfds{chunkMemfds}, _mapBytes{chunkMapBytes} {}

std::expected<IpcHandle, std::string>
ShmIpcExporter::exportChunk(std::uint32_t index, void* /*base*/,
                            std::uint64_t /*usedBytes*/) noexcept {
    if (index >= _memfds.size() || index >= _mapBytes.size()) {
        std::ostringstream os;
        os << "ShmIpcExporter: chunk index " << index << " out of range ("
           << _memfds.size() << " memfd(s), " << _mapBytes.size()
           << " size(s))";
        return std::unexpected(os.str());
    }
    const int fd = _memfds[index];
    if (fd < 0) {
        std::ostringstream os;
        os << "ShmIpcExporter: chunk[" << index << "] has an invalid memfd";
        return std::unexpected(os.str());
    }

    IpcHandle h{};
    // Encode the mmap length little-endian in the first 8 bytes of the
    // 64-byte payload; the worker's ShmIpcImporter reads it back to size
    // its mmap(). The remaining bytes stay zero (reserved).
    const std::uint64_t mapLen = _mapBytes[index];
    std::memcpy(h.bytes.data(), &mapLen, sizeof(mapLen));
    h.fd = fd;  // borrowed — the allocator keeps the memfd open
    return h;
}

} // namespace mimirmind::core::ipc
