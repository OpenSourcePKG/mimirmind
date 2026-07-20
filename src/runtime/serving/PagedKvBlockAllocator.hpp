// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace mimirmind::runtime::serving {

/**
 * Logical block-pool for PagedAttention KV storage. Manages block-id
 * allocation + refcount + per-block hash — the LOGICAL layer only.
 * The PHYSICAL device memory that each block-id refers to is owned by
 * the ComputeContext's allocator and gets wired up in Phase B when
 * `attention_paged_v1.cu` lands.
 *
 * Design lifted from vLLM V1's block-manager + SGLang's hash-chain
 * primitive (see Synaipse `research/paged-attention-implementations-
 * analysis-2026-07-20.md`). The `block_hash: uint64_t` slot is
 * present from Phase A even though the M-PrefixCache consumer lands
 * later — retrofitting the descriptor after the fact would invalidate
 * the on-disk format, the kernel-facing block-table layout, and any
 * Bragi-v1 perf-ledger numbers already gathered.
 *
 * Block-size default = 16 tokens/block (vLLM-consistent). Configurable
 * to 8 or 32 via ctor for A/B-tuning; larger blocks reduce table
 * overhead but increase fragmentation on short sequences.
 *
 * Not thread-safe. The M-Cuda.Batch scheduler owns exactly one
 * `PagedKvBlockAllocator` per model-instance and serialises all
 * allocation calls onto its single-thread event loop.
 *
 * ---- On-disk snapshot format (M-Cuda.Batch Phase A5 spec) ---------
 *
 * When snapshotting a request state (future milestone; format defined
 * now so we don't grow it later), each block descriptor serialises to
 * exactly 16 bytes:
 *
 *   offset 0  : uint32_t physical_id
 *   offset 4  : uint32_t refcount
 *   offset 8  : uint64_t block_hash
 *
 * Endianness = host (Grace ARM = little-endian; if we ever restore on
 * a big-endian host, the serialiser will need to byte-swap).
 *
 * Sequences serialise as:
 *
 *   uint32_t num_blocks
 *   uint32_t num_tokens
 *   int32_t  block_table[num_blocks]     // -1 sentinel for unused slots
 *   int32_t  tokens[num_tokens]          // history for hash-recompute
 *
 * The block-hashes themselves are NOT stored in the sequence-side —
 * they live in the allocator side and get restored via the physical_id
 * lookup on load.
 */
class PagedKvBlockAllocator {
public:
    /// Sentinel returned by `allocate()` when the pool is exhausted.
    /// Callers must check before using the returned id.
    static constexpr std::uint32_t kInvalidBlock =
        std::numeric_limits<std::uint32_t>::max();

    /**
     * Construct a pool of `numBlocks` slots, each of capacity
     * `blockSize` tokens. Neither number can be zero; ctor asserts.
     * All blocks start in the free list.
     */
    PagedKvBlockAllocator(std::size_t numBlocks, std::size_t blockSize);

    PagedKvBlockAllocator(const PagedKvBlockAllocator&)            = delete;
    PagedKvBlockAllocator& operator=(const PagedKvBlockAllocator&) = delete;
    PagedKvBlockAllocator(PagedKvBlockAllocator&&) noexcept        = default;
    PagedKvBlockAllocator& operator=(PagedKvBlockAllocator&&) noexcept = default;

    /**
     * Allocate one block. Returns its physical id (usable as an index
     * into the future KV-memory pool). Sets refcount=1 and clears the
     * block-hash slot. Returns `kInvalidBlock` when the free-list is
     * empty — caller is expected to trigger preemption (M-Cuda.Batch
     * Phase C RECOMPUTE-only policy) and retry.
     *
     * Never throws.
     */
    [[nodiscard]] std::uint32_t allocate() noexcept;

    /**
     * Increment the refcount on an already-allocated block. Used by
     * the future M-PrefixCache consumer to share a common prefix
     * across multiple sequences without duplicating KV memory.
     * No-op for an invalid or free block (logs at WARN in debug
     * builds; silent in release for hot-path safety).
     */
    void addRef(std::uint32_t blockId) noexcept;

    /**
     * Decrement the refcount. When it reaches zero the block goes
     * back on the free list and its hash slot is cleared. Called by
     * `PagedKvSequence::reset()` on RECOMPUTE preemption and by the
     * destructor of a `PagedKvSequence` on request-completion.
     */
    void release(std::uint32_t blockId) noexcept;

    /**
     * Set / read the deterministic block-hash. The M-Cuda.Batch scheduler
     * (or later the M-PrefixCache lookup) computes the hash from the
     * tokens in the fully-filled block plus the parent block's hash
     * (FNV-1a chain — see PagedKvSequence). Present in Phase A even
     * though no consumer reads it yet — see design rationale in the
     * class docstring.
     */
    void          setHash(std::uint32_t blockId, std::uint64_t hash) noexcept;
    [[nodiscard]] std::uint64_t hashOf(std::uint32_t blockId) const noexcept;

    /// Refcount inspection — meant for tests + `/v1/system/status`
    /// metrics (Phase E).
    [[nodiscard]] std::uint32_t refcountOf(std::uint32_t blockId) const noexcept;

    // ---- pool inspection ---------------------------------------------

    [[nodiscard]] std::size_t numBlocksTotal() const noexcept { return _blocks.size(); }
    [[nodiscard]] std::size_t numBlocksFree()  const noexcept { return _freeList.size(); }
    [[nodiscard]] std::size_t numBlocksUsed()  const noexcept {
        return _blocks.size() - _freeList.size();
    }
    [[nodiscard]] std::size_t blockSize()      const noexcept { return _blockSize; }

private:
    struct BlockDescriptor {
        std::uint32_t refcount{0};       // 0 = free
        std::uint64_t hash{0};           // set by scheduler once block is full
    };

    std::size_t                  _blockSize{16};
    std::vector<BlockDescriptor> _blocks;      // indexed by physical id
    std::vector<std::uint32_t>   _freeList;    // LIFO — better cache locality on reuse
};

} // namespace mimirmind::runtime::serving
