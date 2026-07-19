// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/Attention.hpp"

#include "core/log/Log.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mimirmind::compute {

void multiHeadAttention(const float* q,
                        const float* k,
                        const float* v,
                        std::size_t  T_q,
                        std::size_t  T_k,
                        std::size_t  nHeads,
                        std::size_t  nKvHeads,
                        std::size_t  headDim,
                        std::size_t  positionOffset,
                        float*       scratch,
                        float*       out,
                        std::size_t  slidingWindow,
                        float        scale) {
    if (nKvHeads == 0 || nHeads == 0 || nHeads % nKvHeads != 0) {
        throw std::runtime_error(
            "attention: nHeads must be a positive multiple of nKvHeads");
    }
    const std::size_t qStride  = nHeads   * headDim;
    const std::size_t kvStride = nKvHeads * headDim;
    // Sentinel: 0 (or negative) means "use default 1/sqrt(headDim)".
    // Callers pass a positive override (e.g. Gemma 4's 1.0F) when Q
    // was pre-scaled elsewhere in the block. Never zero legitimately
    // in production paths.
    const float effectiveScale = (scale > 0.0F)
        ? scale
        : (1.0F / std::sqrt(static_cast<float>(headDim)));

    for (std::size_t hq = 0; hq < nHeads; ++hq) {
        const std::size_t hkv = (hq * nKvHeads) / nHeads;

        for (std::size_t p = 0; p < T_q; ++p) {
            const std::size_t absPos = positionOffset + p;
            const std::size_t kMax   = std::min(absPos + 1, T_k);
            const std::size_t kMin   =
                (slidingWindow > 0 && kMax > slidingWindow)
                    ? (kMax - slidingWindow) : 0;

            const float* qVec = q + p * qStride + hq * headDim;

            for (std::size_t kk = kMin; kk < kMax; ++kk) {
                const float* kVec = k + kk * kvStride + hkv * headDim;
                double acc = 0.0;
                for (std::size_t d = 0; d < headDim; ++d) {
                    acc += static_cast<double>(qVec[d]) *
                           static_cast<double>(kVec[d]);
                }
                scratch[kk] = static_cast<float>(acc) * effectiveScale;
            }
            for (std::size_t kk = kMax; kk < T_k; ++kk) {
                scratch[kk] = 0.0F;
            }

            // Numerically stable softmax over the live window
            // [kMin, kMax). Slots outside the window stay 0 and never
            // contribute to the V accumulation below.
            float maxScore = -std::numeric_limits<float>::infinity();
            for (std::size_t kk = kMin; kk < kMax; ++kk) {
                if (scratch[kk] > maxScore) maxScore = scratch[kk];
            }
            double sumExp = 0.0;
            for (std::size_t kk = kMin; kk < kMax; ++kk) {
                const float e = std::exp(scratch[kk] - maxScore);
                scratch[kk] = e;
                sumExp += static_cast<double>(e);
            }
            const float invSum = (sumExp > 0.0)
                ? static_cast<float>(1.0 / sumExp) : 0.0F;
            for (std::size_t kk = kMin; kk < kMax; ++kk) {
                scratch[kk] *= invSum;
            }

            float* outVec = out + p * qStride + hq * headDim;
            for (std::size_t d = 0; d < headDim; ++d) {
                double acc = 0.0;
                for (std::size_t kk = kMin; kk < kMax; ++kk) {
                    const float* vVec = v + kk * kvStride + hkv * headDim;
                    acc += static_cast<double>(scratch[kk]) *
                           static_cast<double>(vVec[d]);
                }
                outVec[d] = static_cast<float>(acc);
            }
        }
    }

    MM_LOG_DEBUG("attn",
                 "done — T_q={} T_k={} posOffset={} nHeads={} nKvHeads={} "
                 "headDim={} sw={} (GQA group size {})",
                 T_q, T_k, positionOffset, nHeads, nKvHeads, headDim,
                 slidingWindow, nHeads / nKvHeads);
}

} // namespace mimirmind::compute