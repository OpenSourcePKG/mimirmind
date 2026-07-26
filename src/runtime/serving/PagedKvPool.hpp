// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mimirmind::compute { class ComputeOps; }

namespace mimirmind::runtime::serving {

/**
 * Physical KV storage for PagedAttention (M-Cuda.Batch Phase B/D).
 *
 * `PagedKvBlockAllocator` is the LOGICAL layer — it hands out block ids
 * and refcounts them but owns no device memory. `PagedKvPool` is the
 * PHYSICAL layer those block ids index into: one contiguous K pool and
 * one contiguous V pool per full-attention layer, laid out exactly as
 * the `paged_attention_v1` kernel reads them:
 *
 *   pool[layer] : [numBlocks, blockSize, numKvHeads, headDim]  row-major fp32
 *
 * A global token position `p` of a sequence maps to
 *   blockId = sequence.blockTable()[p / blockSize]
 *   slot    = p % blockSize
 * and its KV head `h` row lives at
 *   pool + ((blockId * blockSize + slot) * numKvHeads + h) * headDim.
 *
 * Only full-attention layers hold KV; the hybrid GatedDeltaNet layers
 * keep an `SsmState` instead and are NOT allocated here. The caller maps
 * a global block index to a dense pool-layer index (mirroring KvCache's
 * `kvSourceLayer` idea) and passes that dense index to the accessors.
 *
 * fp32 baseline only (matches the KvCache F32-only serving path and the
 * paged_attention_v1 baseline body); fp16 / Q8_0 land with the FA2 body.
 *
 * Lifecycle: one pool per model instance, shared by every concurrent
 * sequence — the block allocator arbitrates which blocks each sequence
 * owns, so there is exactly one physical pool, not one-per-sequence
 * (that is the whole point of paging vs. the single-session KvCache).
 */
class PagedKvPool {
public:
    /**
     * Allocate `numLayers` full-attention K/V pools, each of
     * `numBlocks * blockSize` token slots of width `numKvHeads * headDim`
     * fp32 elements. All arguments must be > 0; ctor throws
     * `std::invalid_argument` otherwise.
     */
    PagedKvPool(compute::ComputeOps& ops,
                std::size_t          numLayers,
                std::size_t          numBlocks,
                std::size_t          blockSize,
                std::size_t          numKvHeads,
                std::size_t          headDim);

    PagedKvPool(const PagedKvPool&)            = delete;
    PagedKvPool& operator=(const PagedKvPool&) = delete;
    PagedKvPool(PagedKvPool&&)                 = delete;
    PagedKvPool& operator=(PagedKvPool&&)      = delete;

    /// Base of the K / V pool for dense full-attention layer `layer`.
    /// Pass straight as the kernel's `key_cache` / `value_cache`.
    [[nodiscard]] float*       keyPool(std::size_t layer) noexcept;
    [[nodiscard]] float*       valuePool(std::size_t layer) noexcept;
    [[nodiscard]] const float* keyPool(std::size_t layer) const noexcept;
    [[nodiscard]] const float* valuePool(std::size_t layer) const noexcept;

    /**
     * Append one sequence's freshly-projected K/V row (already QK-normed
     * and RoPE'd for K) into layer `layer` at (`blockId`, `slot`). `kRow`
     * / `vRow` are device pointers of `numKvHeads * headDim` fp32 each.
     * Queued on the compute stream via `ComputeOps::appendMemoryCopy`.
     *
     * The batched decode step calls this once per active sequence after
     * the row-parallel projection + norm + RoPE (the compact per-seq rows
     * scatter into their paged slots here).
     */
    void writeToken(compute::ComputeOps& ops,
                    std::size_t          layer,
                    std::uint32_t        blockId,
                    std::size_t          slot,
                    const float*         kRow,
                    const float*         vRow);

    [[nodiscard]] std::size_t numLayers()  const noexcept { return _numLayers; }
    [[nodiscard]] std::size_t numBlocks()  const noexcept { return _numBlocks; }
    [[nodiscard]] std::size_t blockSize()  const noexcept { return _blockSize; }
    [[nodiscard]] std::size_t numKvHeads() const noexcept { return _numKvHeads; }
    [[nodiscard]] std::size_t headDim()    const noexcept { return _headDim; }

    /// Elements per token slot (one KV head row × all heads) = kv_dim.
    [[nodiscard]] std::size_t slotElems() const noexcept {
        return _numKvHeads * _headDim;
    }
    /// Elements per block = blockSize token slots.
    [[nodiscard]] std::size_t blockElems() const noexcept {
        return _blockSize * slotElems();
    }

private:
    std::size_t _numLayers;
    std::size_t _numBlocks;
    std::size_t _blockSize;
    std::size_t _numKvHeads;
    std::size_t _headDim;

    // One K and one V pool per full-attention layer. Raw pointers cached
    // alongside the RAII owners so the hot path skips the .get() call.
    std::vector<compute::ComputeBuffer> _kOwners;
    std::vector<compute::ComputeBuffer> _vOwners;
    std::vector<float*>                 _kPool;
    std::vector<float*>                 _vPool;
};

} // namespace mimirmind::runtime::serving
