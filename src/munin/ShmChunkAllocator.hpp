// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mimirmind::munin {

/**
 * POSIX-shm (memfd) analogue of ChunkAllocator — the server-side backing
 * store for the M-Munin.CUDA path (ADR 2026-08-14, step 2).
 *
 * Same bump-allocator semantics and layout math as the L0 ChunkAllocator,
 * but each chunk is a `memfd_create` + `ftruncate` + mmap'd MAP_SHARED
 * region of plain host RAM instead of a `zeMemAllocHost` USM allocation.
 * That makes it pure POSIX — no Level Zero, no CUDA — so it lives in
 * mimirmind_core_common and builds/tests on any host. It is what a CUDA
 * Munin uses to hold a model resident: it packs tensors into ~1 GiB chunks
 * and hands each chunk's memfd to ShmIpcExporter, which ships the fd via
 * SCM_RIGHTS; the attached worker mmaps it and dereferences the pointer
 * directly on GB10's SMs (pageableMemAccess + hostPageTables, verified:
 * tools/cuda-ipc-testrig --kind shm).
 *
 * Semantics mirror ChunkAllocator exactly:
 *   - Bump only; no per-allocation free. Chunks released as a whole on
 *     destruction (munmap + close each memfd).
 *   - One allocator per resident model; evict = drop the allocator.
 *   - Each allocation rounded up to `align` (default 64 B).
 *   - A request larger than `chunkBytes` throws — bump does not span chunks.
 *
 * The chunk size cap that motivates a small `chunkBytes` on Xe-LPG (driver
 * `maxMemAllocSize`) does NOT apply here — GB10 unified memory has no such
 * per-allocation ceiling — but keeping the same chunked layout preserves
 * one wire format and one manifest builder across backends.
 *
 * Not thread-safe; model loading is serial at Munin startup.
 */
class ShmChunkAllocator {
public:
    /// Default chunk size. 1 GiB keeps chunk-count modest for multi-GB
    /// models while staying comfortably under any single tensor's size for
    /// the current targets. memfd pages are allocated lazily on first
    /// write, so an unused tail costs no RAM until touched.
    static constexpr std::size_t kDefaultChunkBytes = 1ULL << 30; // 1 GiB

    /// Alignment used when the caller does not request an override.
    static constexpr std::size_t kDefaultAlignment = 64;

    struct Allocation {
        std::uint32_t chunkIndex{0};
        std::uint64_t chunkOffset{0};
        void*         ptr{nullptr};
    };

    /**
     * Construct an allocator drawing fixed-size memfd chunks. `chunkBytes`
     * must be non-zero and a multiple of 4 KiB. The first chunk is opened
     * lazily on the first `allocate`, so an unused allocator costs nothing.
     * Throws std::runtime_error on an invalid `chunkBytes`.
     */
    explicit ShmChunkAllocator(std::size_t chunkBytes = kDefaultChunkBytes);

    ~ShmChunkAllocator();

    ShmChunkAllocator(const ShmChunkAllocator&)            = delete;
    ShmChunkAllocator& operator=(const ShmChunkAllocator&) = delete;
    ShmChunkAllocator(ShmChunkAllocator&&)                 = delete;
    ShmChunkAllocator& operator=(ShmChunkAllocator&&)      = delete;

    /**
     * Reserve `bytes` inside a chunk, rounded up to `align`. Opens a fresh
     * memfd chunk if the current one cannot fit the request. Throws
     * std::runtime_error if `bytes == 0`, `bytes > chunkBytes()`, or a
     * memfd/ftruncate/mmap syscall fails.
     */
    [[nodiscard]] Allocation allocate(std::size_t bytes,
                                      std::size_t align = kDefaultAlignment);

    [[nodiscard]] std::uint32_t chunkCount() const noexcept {
        return static_cast<std::uint32_t>(_chunks.size());
    }

    [[nodiscard]] std::size_t chunkBytes() const noexcept { return _chunkBytes; }

    [[nodiscard]] void* chunkBase(std::uint32_t index) const noexcept {
        return index < _chunks.size() ? _chunks[index].base : nullptr;
    }

    [[nodiscard]] std::uint64_t chunkUsedBytes(std::uint32_t index) const noexcept {
        return index < _chunks.size() ? _chunks[index].used : 0;
    }

    [[nodiscard]] std::uint64_t bytesUsed() const noexcept { return _bytesUsed; }

    /// The memfd backing chunk `index`, or -1 if out of range. Borrowed —
    /// the allocator keeps it open for its lifetime; ShmIpcExporter ships
    /// it via SCM_RIGHTS without closing it.
    [[nodiscard]] int chunkMemfd(std::uint32_t index) const noexcept {
        return index < _chunks.size() ? _chunks[index].memfd : -1;
    }

    /// Snapshot of every chunk's memfd, in chunk order — the exact input to
    /// ShmIpcExporter's `chunkMemfds` span.
    [[nodiscard]] std::vector<int> chunkMemfds() const;

    /// Snapshot of every chunk's mmap length (== chunkBytes each) — the
    /// `chunkMapBytes` span for ShmIpcExporter.
    [[nodiscard]] std::vector<std::uint64_t> chunkMapBytes() const;

    /**
     * Pure bump-placement math, mirrors ChunkAllocator::layoutInsideChunk
     * byte-for-byte (kept separate rather than shared to keep this class
     * free of the L0-tied header). Unit-testable without any syscalls.
     */
    struct Layout {
        bool          needsNewChunk{false};
        std::uint64_t offset{0};
    };

    [[nodiscard]] static Layout layoutInsideChunk(std::uint64_t currentUsed,
                                                  std::size_t   bytes,
                                                  std::size_t   align,
                                                  std::size_t   chunkBytes) noexcept;

private:
    struct Chunk {
        int           memfd{-1};
        void*         base{nullptr};
        std::uint64_t used{0};
    };

    /// memfd_create + ftruncate(chunkBytes) + mmap; push onto `_chunks`.
    /// Throws std::runtime_error on any syscall failure.
    void openChunk();

    std::size_t        _chunkBytes{kDefaultChunkBytes};
    std::vector<Chunk> _chunks{};
    std::uint64_t      _bytesUsed{0};
};

} // namespace mimirmind::munin
