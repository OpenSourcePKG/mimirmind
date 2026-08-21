// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "munin/ShmChunkAllocator.hpp"

#include "core/log/Log.hpp"

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U  // not always exposed by <sys/mman.h> without _GNU_SOURCE
#endif

namespace mimirmind::munin {

namespace {

constexpr std::size_t kMinChunkBytes = 4096;

[[nodiscard]] std::uint64_t alignUp(std::uint64_t value, std::uint64_t align) noexcept {
    if (align <= 1) {
        return value;
    }
    return (value + align - 1) & ~(align - 1);
}

} // namespace

ShmChunkAllocator::ShmChunkAllocator(std::size_t chunkBytes)
    : _chunkBytes(chunkBytes) {
    if (_chunkBytes < kMinChunkBytes) {
        throw std::runtime_error{
            "ShmChunkAllocator: chunkBytes must be at least " +
            std::to_string(kMinChunkBytes) + " (got " +
            std::to_string(_chunkBytes) + ")"};
    }
    if ((_chunkBytes % kMinChunkBytes) != 0) {
        throw std::runtime_error{
            "ShmChunkAllocator: chunkBytes must be a multiple of " +
            std::to_string(kMinChunkBytes) + " (got " +
            std::to_string(_chunkBytes) + ")"};
    }
}

ShmChunkAllocator::~ShmChunkAllocator() {
    for (const auto& c : _chunks) {
        if (c.base != nullptr && c.base != MAP_FAILED) {
            ::munmap(c.base, _chunkBytes);
        }
        if (c.memfd >= 0) {
            ::close(c.memfd);
        }
    }
}

void ShmChunkAllocator::openChunk() {
    const long fd = ::syscall(SYS_memfd_create, "munin-model-chunk", MFD_CLOEXEC);
    if (fd < 0) {
        const int e = errno;
        throw std::runtime_error{
            "ShmChunkAllocator: memfd_create failed: " +
            std::string{std::strerror(e)} + " (errno=" + std::to_string(e) + ")"};
    }
    if (::ftruncate(static_cast<int>(fd), static_cast<off_t>(_chunkBytes)) != 0) {
        const int e = errno;
        ::close(static_cast<int>(fd));
        throw std::runtime_error{
            "ShmChunkAllocator: ftruncate(chunkBytes=" +
            std::to_string(_chunkBytes) + ") failed: " +
            std::string{std::strerror(e)} + " (errno=" + std::to_string(e) + ")"};
    }

    void* base = ::mmap(nullptr, _chunkBytes, PROT_READ | PROT_WRITE,
                        MAP_SHARED, static_cast<int>(fd), 0);
    if (base == MAP_FAILED) {
        const int e = errno;
        ::close(static_cast<int>(fd));
        throw std::runtime_error{
            "ShmChunkAllocator: mmap(chunkBytes=" +
            std::to_string(_chunkBytes) + ") failed: " +
            std::string{std::strerror(e)} + " (errno=" + std::to_string(e) + ")"};
    }

    _chunks.push_back(Chunk{static_cast<int>(fd), base, 0});
    MM_LOG_INFO("munin",
                "ShmChunkAllocator: opened chunk #{} memfd={} base={} size={} bytes",
                _chunks.size() - 1, static_cast<int>(fd), base, _chunkBytes);
}

ShmChunkAllocator::Layout
ShmChunkAllocator::layoutInsideChunk(std::uint64_t currentUsed,
                                     std::size_t   bytes,
                                     std::size_t   align,
                                     std::size_t   chunkBytes) noexcept {
    const std::uint64_t effectiveAlign =
        (align < 1) ? std::uint64_t{1} : static_cast<std::uint64_t>(align);
    const std::uint64_t offset = alignUp(currentUsed, effectiveAlign);
    if (offset + static_cast<std::uint64_t>(bytes) > chunkBytes) {
        return Layout{true, 0};
    }
    return Layout{false, offset};
}

ShmChunkAllocator::Allocation
ShmChunkAllocator::allocate(std::size_t bytes, std::size_t align) {
    if (bytes == 0) {
        throw std::runtime_error{"ShmChunkAllocator::allocate: bytes == 0"};
    }
    if (bytes > _chunkBytes) {
        throw std::runtime_error{
            "ShmChunkAllocator::allocate: request " + std::to_string(bytes) +
            " exceeds chunkBytes " + std::to_string(_chunkBytes) +
            " — raise chunkBytes or split the tensor"};
    }

    const std::size_t effectiveAlign = align < 1 ? kDefaultAlignment : align;

    if (_chunks.empty()) {
        openChunk();
    }

    Chunk* current = &_chunks.back();
    Layout plan    = layoutInsideChunk(current->used, bytes, effectiveAlign,
                                       _chunkBytes);
    if (plan.needsNewChunk) {
        openChunk();
        current = &_chunks.back();
        plan    = layoutInsideChunk(current->used, bytes, effectiveAlign,
                                    _chunkBytes);
        if (plan.needsNewChunk) {
            // Only reachable if bytes > chunkBytes, already refused above.
            throw std::runtime_error{
                "ShmChunkAllocator: fresh chunk cannot fit request " +
                std::to_string(bytes) + " after align " +
                std::to_string(effectiveAlign) + " — invariant violated"};
        }
    }

    Allocation out{};
    out.chunkIndex  = static_cast<std::uint32_t>(_chunks.size() - 1);
    out.chunkOffset = plan.offset;
    out.ptr         = static_cast<std::byte*>(current->base) + plan.offset;

    current->used = plan.offset + bytes;
    _bytesUsed += bytes;

    return out;
}

std::vector<int> ShmChunkAllocator::chunkMemfds() const {
    std::vector<int> fds;
    fds.reserve(_chunks.size());
    for (const auto& c : _chunks) {
        fds.push_back(c.memfd);
    }
    return fds;
}

std::vector<std::uint64_t> ShmChunkAllocator::chunkMapBytes() const {
    std::vector<std::uint64_t> sizes;
    sizes.reserve(_chunks.size());
    for (std::size_t i = 0; i < _chunks.size(); ++i) {
        sizes.push_back(static_cast<std::uint64_t>(_chunkBytes));
    }
    return sizes;
}

} // namespace mimirmind::munin
