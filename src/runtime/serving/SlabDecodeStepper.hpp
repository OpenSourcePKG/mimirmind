// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mimirmind::compute {
class ComputeOps;
class ComputeMatmul;
} // namespace mimirmind::compute

namespace mimirmind::core::gguf { struct GgufTensor; }

namespace mimirmind::runtime {
class KvCache;
struct BlockBuffers;
} // namespace mimirmind::runtime

namespace mimirmind::runtime::arch { class ArchBackend; }

namespace mimirmind::runtime::serving {

class KvCacheSlabPool;

/**
 * Synchronized batched decode step for the L0 / Xe-LPG serving path
 * (M9.1 / ADR 2026-08-18 Step 3).
 *
 * One `step()` decodes ONE token for each of the active `nSeq` slots in
 * lock-step, driving the backend-neutral
 * `ArchBackend::runBlockBatched(std::span<KvCache* const>)` over a
 * `KvCacheSlabPool`. Each slot's slab `KvCache` sits at its OWN length
 * (== its decode position), so the substrate needs no shared block table
 * or paged pool — the non-paged slab pool is exactly the per-sequence
 * `KvCache` contract `runBlockBatched` already writes into.
 *
 * Self-contained: it depends only on the ComputeOps / ComputeMatmul /
 * ArchBackend interfaces + the slab pool, NOT on InferenceEngine, so the
 * serving substrate can be unit-tested and reasoned about in isolation.
 * The engine constructs it once per serving engine and feeds it the model
 * weight tensors + dims.
 *
 * F32 logits baseline; greedy argmax per slot (the serving parity gate,
 * matching the qwen35moe `ServingSession::stepServing` sampler).
 */
class SlabDecodeStepper {
public:
    /// Model weight tensors the decode step reads (owned by the engine).
    struct Weights {
        const core::gguf::GgufTensor* tokEmb;   ///< token embedding table
        const core::gguf::GgufTensor* outNorm;  ///< final RMSNorm weight
        const core::gguf::GgufTensor* lmHead;   ///< output projection
    };

    /// Scalar model dims + flags.
    struct Dims {
        std::size_t dModel;
        std::size_t vocabLm;
        std::size_t vocabEmb;
        std::size_t blockCount;
        float       rmsNormEps;
        bool        scaleEmbedding;  ///< Gemma: scale embed by sqrt(dModel)
    };

    /// Allocates decode scratch for up to `slabPool.capacity()` rows.
    SlabDecodeStepper(compute::ComputeOps&      ops,
                      compute::ComputeMatmul&   gmm,
                      arch::ArchBackend&        backend,
                      KvCacheSlabPool&          slabPool,
                      const Weights&            weights,
                      const Dims&               dims);

    SlabDecodeStepper(const SlabDecodeStepper&)            = delete;
    SlabDecodeStepper& operator=(const SlabDecodeStepper&) = delete;
    SlabDecodeStepper(SlabDecodeStepper&&)                 = delete;
    SlabDecodeStepper& operator=(SlabDecodeStepper&&)      = delete;

    /**
     * One synchronized batched decode step over the active slot prefix
     * `[0, tokens.size())`. `tokens[i]` is slot i's input token; slot i's
     * slab `KvCache` MUST already sit at that slot's current length (its
     * position) — the caller prefills / advances it. Writes the greedy
     * argmax next token per slot into `outTokens` and advances (commits
     * one token into) each active slab.
     *
     * `sb` is caller-owned block scratch sized for `maxT >= capacity()`.
     * Throws if `tokens.size() > capacity()` or `outTokens.size() != nSeq`.
     */
    void step(std::span<const std::int32_t> tokens,
              std::span<std::int32_t>       outTokens,
              BlockBuffers&                 sb);

    [[nodiscard]] std::size_t capacity() const noexcept { return _capacity; }

private:
    compute::ComputeOps&    _ops;
    compute::ComputeMatmul& _gmm;
    arch::ArchBackend&      _backend;
    KvCacheSlabPool&        _slab;
    Weights                 _w;
    Dims                    _dims;
    float                   _embedScale;   ///< sqrt(dModel) or 1.0
    std::size_t             _capacity;

    // Decode scratch (device), sized for `_capacity` rows.
    compute::ComputeBuffer _xBuf;    // [capacity, dModel]
    compute::ComputeBuffer _normBuf; // [capacity, dModel]
    compute::ComputeBuffer _logits;  // [capacity, vocabLm]
    compute::ComputeBuffer _lmScr;   // [max(dModel, vocabLm)]

    // Host staging.
    std::vector<float>    _hostLogits; // [capacity * vocabLm]
    std::vector<KvCache*> _caches;     // reusable active-slot span backing
};

} // namespace mimirmind::runtime::serving
