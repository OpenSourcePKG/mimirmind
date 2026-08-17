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
                                     const Dims&             dims)
    : _ops(ops),
      _gmm(gmm),
      _backend(backend),
      _slab(slabPool),
      _w(weights),
      _dims(dims),
      _embedScale(dims.scaleEmbedding
                      ? std::sqrt(static_cast<float>(dims.dModel))
                      : 1.0F),
      _capacity(slabPool.capacity())
{
    if (_capacity == 0) {
        throw std::invalid_argument(
            "SlabDecodeStepper: slab pool capacity must be > 0");
    }
    if (_w.tokEmb == nullptr || _w.outNorm == nullptr || _w.lmHead == nullptr) {
        throw std::invalid_argument(
            "SlabDecodeStepper: weight tensors must be non-null");
    }
    if (_dims.dModel == 0 || _dims.vocabLm == 0 || _dims.blockCount == 0) {
        throw std::invalid_argument(
            "SlabDecodeStepper: dModel / vocabLm / blockCount must be > 0");
    }

    _xBuf    = _ops.allocate(_capacity * _dims.dModel  * sizeof(float));
    _normBuf = _ops.allocate(_capacity * _dims.dModel  * sizeof(float));
    _logits  = _ops.allocate(_capacity * _dims.vocabLm * sizeof(float));
    _lmScr   = _ops.allocate(std::max(_dims.dModel, _dims.vocabLm) * sizeof(float));
    _hostLogits.resize(_capacity * _dims.vocabLm);
    _caches.reserve(_capacity);
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
