// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/serving/PagedKvBlockAllocator.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mimirmind::runtime::serving {

/**
 * Per-request page-table + token history for PagedAttention KV storage.
 * Sits above `PagedKvBlockAllocator` (the LOGICAL block pool) and gets
 * consumed later by the Phase-B `attention_paged_v1.cu` kernel through
 * a `blockTable()`-array kernel arg.
 *
 * Ownership model: the M-Cuda.Batch scheduler holds one
 * `PagedKvSequence` per active request. Blocks are borrowed from a
 * shared `PagedKvBlockAllocator` (passed by reference in the ctor); the
 * sequence returns them via `release()` in the allocator on `reset()`
 * or destruction.
 *
 * Not thread-safe — same rationale as the allocator: the M-Cuda.Batch
 * scheduler serialises all mutations onto its single-thread event loop.
 *
 * ---- Block-hash chain (FNV-1a-64) ---------------------------------
 *
 * When a block becomes full (last token slot occupied), the sequence
 * computes a deterministic FNV-1a-64 hash over the block's tokens
 * seeded with the *parent* block's hash. For block 0 the seed is the
 * FNV-1a offset basis. The hash is written back into the allocator via
 * `setHash()` so the future M-PrefixCache consumer can look up shared
 * prefixes across sequences.
 *
 * Deterministic property (relied on by prefix-cache lookup):
 * two sequences that append the same first `K * blockSize` tokens
 * produce the same block-hashes for blocks 0..K-1.
 *
 * ---- Serialisation (skeleton, no implementation yet) --------------
 *
 * Wire format is spec'd in `PagedKvBlockAllocator.hpp` — the sequence
 * side emits `num_blocks + num_tokens + block_table[] + tokens[]`.
 * `tokens()` is kept as `int32_t` history to allow full hash-chain
 * recompute on load without persisting the hashes themselves.
 */
class PagedKvSequence {
public:
    /**
     * Bind to an existing allocator. The allocator MUST outlive this
     * sequence — no ownership transfer. Ctor is O(1) and never throws.
     */
    explicit PagedKvSequence(PagedKvBlockAllocator& allocator) noexcept;

    /**
     * Releases every block held by this sequence back to the pool.
     * Safe to call on a moved-from instance.
     */
    ~PagedKvSequence();

    PagedKvSequence(const PagedKvSequence&)            = delete;
    PagedKvSequence& operator=(const PagedKvSequence&) = delete;

    /**
     * Move transfers block ownership. The moved-from instance ends up
     * in the empty state (no allocator ref, no blocks, no tokens) and
     * its destructor becomes a no-op.
     */
    PagedKvSequence(PagedKvSequence&& other) noexcept;
    PagedKvSequence& operator=(PagedKvSequence&& other) noexcept;

    /**
     * Append one token to the sequence. Allocates a new block from the
     * pool when crossing a block boundary. When the append fills the
     * current block, computes the FNV-1a-64 block-hash and writes it
     * into the allocator.
     *
     * Returns true on success. Returns false on pool exhaustion — in
     * which case the sequence state is **unchanged** (no partial token
     * append, no dangling block).
     *
     * Never throws.
     */
    [[nodiscard]] bool appendToken(std::int32_t tokenId) noexcept;

    /**
     * Release every block back to the pool and clear both the block-
     * table and the token history. Called by the M-Cuda.Batch scheduler
     * on RECOMPUTE preemption (the sequence gets rebuilt from the
     * prompt tokens); also called implicitly by the destructor.
     */
    void reset() noexcept;

    // ---- inspection --------------------------------------------------

    [[nodiscard]] std::size_t numTokens() const noexcept { return _tokens.size(); }
    [[nodiscard]] std::size_t numBlocks() const noexcept { return _blockTable.size(); }
    [[nodiscard]] std::size_t blockSize() const noexcept {
        return _allocator ? _allocator->blockSize() : 0;
    }

    [[nodiscard]] const std::vector<std::uint32_t>& blockTable() const noexcept {
        return _blockTable;
    }
    [[nodiscard]] const std::vector<std::int32_t>& tokens() const noexcept {
        return _tokens;
    }

    /**
     * FNV-1a-64 offset basis — exposed as a static constant so tests
     * and the future prefix-cache consumer can compute expected hashes
     * without re-deriving the seed convention.
     */
    static constexpr std::uint64_t kFnvBasis = 0xcbf29ce484222325ULL;

private:
    void releaseAllBlocks() noexcept;

    /**
     * Fold `count` int32 tokens into an FNV-1a-64 hash, seeded with
     * `seed`. Tokens are consumed little-endian byte-by-byte so hash
     * output is stable across compilers (all our target hosts are
     * little-endian; see allocator docstring).
     */
    static std::uint64_t fnv1aChain(std::uint64_t        seed,
                                    const std::int32_t*  data,
                                    std::size_t          count) noexcept;

    PagedKvBlockAllocator*     _allocator{nullptr};
    std::vector<std::uint32_t> _blockTable;
    std::vector<std::int32_t>  _tokens;
};

} // namespace mimirmind::runtime::serving