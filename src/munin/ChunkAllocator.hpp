// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/l0/L0Context.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mimirmind::munin {

/**
 * Bump-allocator over a fixed sequence of large `zeMemAllocHost` chunks.
 *
 * Motivation: Munin exports one L0 IPC handle per USM allocation via
 * `IpcExporter::exportOne`. The Xe-LPG driver on Meteor Lake caps the
 * IPC-handle table at roughly 1024 entries per process, which is
 * exceeded once two mid-size Gemma-4 models are co-resident (658 + 720
 * tensors = 1378 handles). The single big allocation per tensor turns
 * that budget into hard scaling limit at N=1 model.
 *
 * The ChunkAllocator collapses per-tensor allocations into a handful of
 * large chunks (default 1 GiB each, configurable). Each tensor lives at
 * `(chunkIndex, offset)` inside a chunk. The IPC-Export path then emits
 * one handle per chunk (~30 for two full models) instead of one per
 * tensor — orders of magnitude under the driver cap, with headroom
 * for 3–4 co-resident models.
 *
 * Semantics:
 *   - **Bump only.** Allocations move a monotonic cursor. There is no
 *     per-allocation free — chunks are released as a whole when the
 *     ChunkAllocator is destroyed.
 *   - **Per-model ownership.** One ChunkAllocator per LoadedModel. Model
 *     evict = drop the ChunkAllocator = release the chunks.
 *   - **Alignment.** Each allocation is rounded up to `align` (default
 *     64 B, matching Xe-LPG SIMD16 sub-group + SLM alignment). GGUF's
 *     32-byte block alignment is respected as a lower bound.
 *   - **Chunk overflow.** If the current chunk cannot fit the request,
 *     a new chunk is allocated and the old one is sealed (its unused
 *     tail bytes are dead — bump-allocator wastes tail space by design).
 *   - **Single-alloc cap.** A request larger than `chunkBytes` throws
 *     — bump does not span chunks. Configure `chunkBytes` above the
 *     largest expected tensor (e.g. `token_embd.weight` for Gemma-4 is
 *     ~605 MiB Q6_K, fits comfortably under the 1 GiB default).
 *   - **Driver cap.** `chunkBytes` must stay under Xe-LPG's
 *     `maxMemAllocSize` (~3.94 GiB with relaxed-allocation env vars).
 *     Enforced at construction; throws L0Error on violation only
 *     when the first alloc actually runs (the ceiling is a driver
 *     property, not something we can check upfront cheaply).
 *
 * Not thread-safe. Model loading is serial at Munin startup, and the
 * allocator is not touched after `ModelStore` finishes constructing.
 *
 * See `decisions/2026-07-13-m-munin-chunk-layout.md` (M-Munin.1a ADR)
 * for the full design rationale, wire-format bump, and umsetzungs-
 * reihenfolge.
 */
class ChunkAllocator {
public:
    /// Default chunk size. Sits under Xe-LPG's ~3.94 GiB
    /// `maxMemAllocSize` cap while being big enough to hold Gemma-4's
    /// largest single tensor: E4B `token_embd.weight` is F16 at
    /// 262144 vocab × 4096 dim = 2.15 GiB, so 1 GiB is not enough for
    /// every model. 3 GiB gives ~40% headroom over the observed max
    /// and cuts chunk-count (and therefore Munin boot time) by ~3×
    /// compared to a 1 GiB default. Raise only if a future model has
    /// a single tensor bigger than 3 GiB; requires re-checking against
    /// `maxMemAllocSize` on that hardware.
    static constexpr std::size_t kDefaultChunkBytes = 3ULL << 30; // 3 GiB

    /// Alignment used when the caller does not request an override.
    /// 64 B matches Xe-LPG SIMD16 sub-group loads and the SLM tile
    /// granularity our kernels expect.
    static constexpr std::size_t kDefaultAlignment = 64;

    /**
     * Result of an `allocate` call. `ptr` is the raw USM pointer inside
     * the owning chunk — the caller uses it to `memcpy` GGUF payload in.
     * `chunkIndex` + `chunkOffset` are what the TensorManifest carries
     * over the wire so the attaching worker can reconstruct the pointer
     * from its own imported chunk bases.
     */
    struct Allocation {
        std::uint32_t chunkIndex{0};
        std::uint64_t chunkOffset{0};
        void*         ptr{nullptr};
    };

    /**
     * Construct an allocator that will draw fixed-size chunks from `ctx`
     * via `zeMemAllocHost`. `chunkBytes` is validated: must be non-zero
     * and a positive multiple of 4 KiB. The first chunk is not
     * allocated eagerly — `allocate` triggers the first chunk lazily so
     * an unused ChunkAllocator costs no host RAM.
     */
    ChunkAllocator(::mimirmind::core::l0::L0Context& ctx,
                   std::size_t                       chunkBytes = kDefaultChunkBytes);

    ~ChunkAllocator();

    ChunkAllocator(const ChunkAllocator&)            = delete;
    ChunkAllocator& operator=(const ChunkAllocator&) = delete;
    ChunkAllocator(ChunkAllocator&&)                 = delete;
    ChunkAllocator& operator=(ChunkAllocator&&)      = delete;

    /**
     * Reserve `bytes` inside a chunk, rounded up to `align`. Opens a
     * new chunk if the current one cannot fit the request. Throws
     * `std::runtime_error` if `bytes > chunkBytes()` (single alloc
     * cannot span chunks) or `L0Error` when the driver refuses a new
     * chunk allocation.
     */
    [[nodiscard]] Allocation allocate(std::size_t bytes,
                                      std::size_t align = kDefaultAlignment);

    /// Number of chunks allocated so far. Zero before the first
    /// `allocate` call.
    [[nodiscard]] std::uint32_t chunkCount() const noexcept {
        return static_cast<std::uint32_t>(_chunks.size());
    }

    /// The configured chunk size in bytes.
    [[nodiscard]] std::size_t chunkBytes() const noexcept { return _chunkBytes; }

    /// Base USM pointer of chunk `index`. Consumed by the IPC-Export
    /// path when it walks chunks to produce handles.
    [[nodiscard]] void* chunkBase(std::uint32_t index) const noexcept {
        return index < _chunks.size() ? _chunks[index].base : nullptr;
    }

    /// Actual bytes used inside chunk `index`. Munin sends this as the
    /// `ChunkDesc.bytes` field so the worker can sanity-check its
    /// imported chunk's declared size against the payload footprint.
    [[nodiscard]] std::uint64_t chunkUsedBytes(std::uint32_t index) const noexcept {
        return index < _chunks.size() ? _chunks[index].used : 0;
    }

    /// Sum of bytes served across all chunks. Excludes tail waste
    /// inside sealed chunks; add `(chunkCount() * chunkBytes()) -
    /// bytesUsed()` to see how much is padding / dead-tail.
    [[nodiscard]] std::uint64_t bytesUsed() const noexcept { return _bytesUsed; }

    /**
     * Pure decision function extracted from `allocate` so the bump
     * math is unit-testable without a live L0 device. Given the
     * currently-used bytes inside a chunk plus the request, decide
     * whether the request fits into the current chunk (in which case
     * `offset` is the aligned placement) or whether a fresh chunk
     * must be opened first (in which case `needsNewChunk` is true and
     * `offset` is the placement inside the fresh chunk, which is
     * simply `alignUp(0, align)` = 0 for any sane alignment).
     *
     * Preconditions the caller must have already checked:
     *   - `bytes > 0`
     *   - `bytes <= chunkBytes`
     *
     * `align` is treated as at least 1 (0 is silently clamped up).
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
        void*         base{nullptr};
        std::uint64_t used{0};
    };

    /// Allocate a fresh chunk from the driver and push it onto `_chunks`.
    /// Throws L0Error on driver failure.
    void openChunk();

    ::mimirmind::core::l0::L0Context& _ctx;
    std::size_t                       _chunkBytes{kDefaultChunkBytes};
    std::vector<Chunk>                _chunks{};
    std::uint64_t                     _bytesUsed{0};
};

} // namespace mimirmind::munin