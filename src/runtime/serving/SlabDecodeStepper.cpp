// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/SlabDecodeStepper.hpp"

#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "compute/Embedding.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/arch/ArchBackend.hpp"
#include "runtime/serving/KvCacheSlabPool.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::serving {

SlabDecodeStepper::SlabDecodeStepper(compute::ComputeOps&    ops,
                                     compute::ComputeMatmul& gmm,
                                     arch::ArchBackend&      backend,
                                     KvCacheSlabPool&        slabPool,
                                     const Weights&          weights,
                                     const Dims&             dims,
                                     std::size_t             maxPrefillT)
    : _ops(ops),
      _gmm(gmm),
      _backend(backend),
      _slab(slabPool),
      _w(weights),
      _dims(dims),
      _embedScale(dims.scaleEmbedding
                      ? std::sqrt(static_cast<float>(dims.dModel))
                      : 1.0F),
      _capacity(slabPool.capacity()),
      _maxPrefillT(maxPrefillT)
{
    if (_capacity == 0) {
        throw std::invalid_argument(
            "SlabDecodeStepper: slab pool capacity must be > 0");
    }
    if (_maxPrefillT == 0) {
        throw std::invalid_argument(
            "SlabDecodeStepper: maxPrefillT must be > 0");
    }
    if (_w.tokEmb == nullptr || _w.outNorm == nullptr || _w.lmHead == nullptr) {
        throw std::invalid_argument(
            "SlabDecodeStepper: weight tensors must be non-null");
    }
    if (_dims.dModel == 0 || _dims.vocabLm == 0 || _dims.blockCount == 0) {
        throw std::invalid_argument(
            "SlabDecodeStepper: dModel / vocabLm / blockCount must be > 0");
    }

    // Decode scratch is sized for the wider of the decode batch (`_capacity`
    // rows) and one prefill chunk (`_maxPrefillT` rows) so the shared final-
    // norm / lm-head buffers cover the prefill first-token sample too.
    const std::size_t maxRows = std::max(_capacity, _maxPrefillT);
    _xBuf    = _ops.allocate(_capacity   * _dims.dModel  * sizeof(float));
    _normBuf = _ops.allocate(maxRows     * _dims.dModel  * sizeof(float));
    _logits  = _ops.allocate(maxRows     * _dims.vocabLm * sizeof(float));
    _lmScr   = _ops.allocate(std::max(_dims.dModel, _dims.vocabLm) * sizeof(float));
    _xBufP   = _ops.allocate(_maxPrefillT * _dims.dModel * sizeof(float));
    _hostLogits.resize(maxRows * _dims.vocabLm);
    _caches.reserve(_capacity);
}

std::int32_t SlabDecodeStepper::prefillSlot(
        std::size_t                   slot,
        std::span<const std::int32_t> tokens,
        BlockBuffers&                 prefillSb,
        bool                          produceToken) {
    const std::size_t T = tokens.size();
    if (T == 0) {
        return -1;
    }
    if (slot >= _capacity) {
        throw std::out_of_range(
            "SlabDecodeStepper::prefillSlot: slot " + std::to_string(slot) +
            " >= capacity " + std::to_string(_capacity));
    }
    if (T > _maxPrefillT) {
        throw std::invalid_argument(
            "SlabDecodeStepper::prefillSlot: chunk " + std::to_string(T) +
            " > maxPrefillT " + std::to_string(_maxPrefillT));
    }
    KvCache& cache = _slab.slot(slot);
    if (cache.length() + T > cache.maxSeq()) {
        throw std::out_of_range(
            "SlabDecodeStepper::prefillSlot: chunk overflows slab context cap");
    }

    float* const xBufP = _xBufP.as<float>();
    compute::embeddingLookup(_w.tokEmb->type, _w.tokEmb->usmPtr,
                             _dims.dModel, _dims.vocabEmb, tokens, xBufP);
    if (_dims.scaleEmbedding) {
        _ops.mulScalarAsync(xBufP, _embedScale, T * _dims.dModel);
    }
    _backend.prepareForward(tokens, xBufP, T);

    // Single-sequence T>1 forward into this slot's slab, appending at the
    // slab's current length. Drain per block on long prompts (the Xe-LPG
    // long-prefill safety cadence — mirrors InferenceEngine's prefill).
    constexpr std::size_t kPrefillDrainThreshold = 256;
    const bool drainPerBlock = T > kPrefillDrainThreshold;
    for (std::size_t b = 0; b < _dims.blockCount; ++b) {
        _backend.runBlock(b, xBufP, T, cache, prefillSb, /*traceBlock0=*/false);
        if (drainPerBlock) {
            _ops.flush();
        }
    }
    cache.commit(T);

    if (!produceToken) {
        return -1;
    }

    // Greedy first token from the last prompt row.
    float* const lastRow = xBufP + (T - 1) * _dims.dModel;
    float* const normBuf = _normBuf.as<float>();
    float* const logits  = _logits.as<float>();
    _ops.rmsNormAsync(lastRow, 1, _dims.dModel,
                      static_cast<const float*>(_w.outNorm->usmPtr),
                      _dims.rmsNormEps, normBuf);
    _gmm.matmul(_w.lmHead->type, _w.lmHead->usmPtr, _dims.vocabLm, _dims.dModel,
                normBuf, 1, logits, _lmScr.as<float>());
    _ops.flush();
    _ops.readbackToHost(_hostLogits.data(), logits, _dims.vocabLm * sizeof(float));
    std::size_t best = 0;
    float       bv   = _hostLogits[0];
    for (std::size_t v = 1; v < _dims.vocabLm; ++v) {
        if (_hostLogits[v] > bv) {
            bv   = _hostLogits[v];
            best = v;
        }
    }
    return static_cast<std::int32_t>(best);
}

void SlabDecodeStepper::step(std::span<const std::int32_t> tokens,
                             std::span<std::int32_t>       outTokens,
                             BlockBuffers&                 sb) {
    const std::size_t nSeq = tokens.size();
    if (nSeq == 0) {
        return;
    }
    if (nSeq > _capacity) {
        throw std::out_of_range(
            "SlabDecodeStepper::step: nSeq " + std::to_string(nSeq) +
            " > capacity " + std::to_string(_capacity));
    }
    if (outTokens.size() != nSeq) {
        throw std::invalid_argument(
            "SlabDecodeStepper::step: outTokens size != tokens size");
    }

    float* const xBuf   = _xBuf.as<float>();
    float* const normBuf = _normBuf.as<float>();
    float* const logits = _logits.as<float>();

    // Row-parallel embedding lookup for the nSeq input tokens.
    compute::embeddingLookup(_w.tokEmb->type, _w.tokEmb->usmPtr,
                             _dims.dModel, _dims.vocabEmb, tokens, xBuf);
    if (_dims.scaleEmbedding) {
        _ops.mulScalarAsync(xBuf, _embedScale, nSeq * _dims.dModel);
    }

    // Per-slot KvCache span for the active prefix. Each slab sits at its
    // own length (== its decode position), so runBlockBatched writes the
    // new K/V row at that length and attention reads [0, length()+1).
    _slab.activeSlotCaches(nSeq, _caches);
    const std::span<KvCache* const> caches{_caches};

    for (std::size_t b = 0; b < _dims.blockCount; ++b) {
        _backend.runBlockBatched(b, xBuf, nSeq, caches, sb, /*diag=*/false);
    }

    // Advance every active slab by one token — the runBlockBatched contract
    // commits each cache once after the whole block chain.
    for (std::size_t i = 0; i < nSeq; ++i) {
        _caches[i]->commit(1);
    }

    // Final norm + LM head → per-slot logits.
    _ops.profileSection("lmhead");
    _ops.rmsNormAsync(xBuf, nSeq, _dims.dModel,
                      static_cast<const float*>(_w.outNorm->usmPtr),
                      _dims.rmsNormEps, normBuf);
    _gmm.matmul(_w.lmHead->type, _w.lmHead->usmPtr, _dims.vocabLm, _dims.dModel,
                normBuf, nSeq, logits, _lmScr.as<float>());
    _ops.profileStepEnd();
    _ops.flush();
    _ops.readbackToHost(_hostLogits.data(), logits,
                        nSeq * _dims.vocabLm * sizeof(float));

    // Greedy argmax per slot (serving parity gate).
    for (std::size_t i = 0; i < nSeq; ++i) {
        const float* row = _hostLogits.data() + i * _dims.vocabLm;
        std::size_t  best = 0;
        float        bv   = row[0];
        for (std::size_t v = 1; v < _dims.vocabLm; ++v) {
            if (row[v] > bv) {
                bv   = row[v];
                best = v;
            }
        }
        outTokens[i] = static_cast<std::int32_t>(best);
    }
}

} // namespace mimirmind::runtime::serving
